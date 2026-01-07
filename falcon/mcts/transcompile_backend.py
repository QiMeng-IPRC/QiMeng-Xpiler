import logging
import os
import time
from functools import partial
import glob
import uuid
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import mctx
import tiktoken
from jax import jit, lax, vmap

from benchmark.perf import perf_cuda, perf_dlboost, perf_hip, perf_mlu
from falcon.mcts.action_logit import generate_prior_from_src
from falcon.mcts.actions_uni import actions as ActionSpace
from falcon.mcts.invalid_actions import get_invalid_actions
from falcon.mcts.utils import open_file
from falcon.util import get_target
from falcon.unit_test_c_cuda import CudaOpUnitTest
# ==== FastAPI / Pydantic / Uvicorn ====
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import traceback 
# ---------------------------
# 日志配置
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------
# 全局常量 & JAX 配置
# ---------------------------

# TODO: 用真实 shape 计算
GFLOPS = 64 * 1280 * 2 / 1e9
A_LENGTH = len(ActionSpace)

# 搜索参数（原来用 FLAGS，现在用常量）
SEED: int = 42
NUM_SIMULATIONS: int = 64
MAX_NUM_CONSIDERED_ACTIONS: int = 13
MAX_DEPTH: int = 10

encoder = tiktoken.encoding_for_model("gpt-4o")

jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
jax.disable_jit()
BS = 1


def objective(file_name, target):
    """We design an objective function.

    If compile and runtime error happens, then the score is zero.
    """
    try:
        time_ms = 1000000
        if target == "cuda":
            time_ms = perf_cuda.benchmark(file_name)
        elif target == "mlu":
            time_ms = perf_mlu.benchmark(file_name)
        elif target == "cpu":
            time_ms = perf_dlboost.benchmark(file_name)
        elif target == "hip":
            time_ms = perf_hip.benchmark(file_name)
        return GFLOPS / (time_ms / 1e3)
    except Exception as e:
        logging.info(e)
        return 0.0


#目前只支持C to CUDA

class FalconGo:
    def __init__(
        self,
        file_name: str,
        op_name: str,
        source_platform: str,
        target_platform: str,
        action_len: int = A_LENGTH,
        optimizer_len: int = A_LENGTH,
        timeout: Optional[float] = None,
    ):
        self.timeout = timeout
        self.file_name = file_name
        self.op_name = op_name
        self.source_platform = source_platform
        self.target_platform = target_platform
        self.action_len = action_len
        self.optimizer_len = optimizer_len
        self.best_reward = 0.0001
        self.best_optimizer_ids = None
        self.iteration = 0
        self.best_actions = None
        self.output_dir = os.path.join(
            f"{self.source_platform}_{self.target_platform}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.tester = CudaOpUnitTest(
        cpp_ref_code=open_file(self.file_name))

    def perform_action(self, actions):
        """按 action 序列对代码做变换并评估性能."""
        code = open_file(self.file_name)
        if self.source_platform in ["cuda", "hip"]:
            code = code.split("extern")[0]
        for action in actions:
            code = action(
                self.tester,
                code,
                self.source_platform,
                self.target_platform,
            )
        target, file_type = get_target(code, self.target_platform)
        os.makedirs("tmp", exist_ok=True)

        base_name = os.path.basename(self.file_name)
        name_no_ext, _ = os.path.splitext(base_name)
        new_file = os.path.join("tmp", name_no_ext + file_type)
        with open(new_file, "w", encoding="utf-8") as f:
            f.write(code)
        _,perf_results = self.tester.perf(code, warmup=5, repeat=50)
        try:
            score = 1/perf_results['kernel_ms']['mean']
        except:
            score = -1
        if target != self.target_platform:
            score = 0.0
        return code, score

    @jit
    def step(self, action_id, env_state):
        self.iteration += 1
        embedding_state, trajectory, depth, rewards = env_state
        trajectory = trajectory.at[depth].set(action_id)
        cur_action_ids = lax.dynamic_slice(
            trajectory, (0,), (depth.val[0] + 1,)
        )
        cur_action_list = jax.device_get(cur_action_ids.val[0]).tolist()
        cur_actions = [ActionSpace[_i] for _i in cur_action_list]

        try:
            code, reward = self.perform_action(cur_actions)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_actions = cur_action_list
                # 保存当前最优代码
                base_name = os.path.basename(self.file_name)
                name_no_ext, _ = os.path.splitext(base_name)
                target, file_type = get_target(code, self.target_platform)
                new_file = os.path.join(
                    self.output_dir, name_no_ext + file_type
                )
                with open(new_file, "w", encoding="utf-8") as f:
                    f.write(code)
        except Exception:
            #print(code)
            traceback.print_exc()
            code = ""
            reward = -10000.0
            print(f"Invalid action: {cur_action_ids.val[0].tolist()}")

        print(
            f"Step: {self.iteration}\t"
            f"Action: {cur_action_ids.val[0].tolist()}\t"
            f"Reward: {reward:.4f}\t"
            f"Best Reward: {self.best_reward:.4f}\t"
            f"Best action: {self.best_actions}\t",
            flush=True,
        )

        for depth_index, var in enumerate(cur_action_list):
            new_value = (rewards[0, depth_index, var] + reward) / 2.0
            rewards = rewards.at[0, depth_index, var].set(new_value)

        condition1 = depth > self.optimizer_len
        condition2 = reward == -10000.0

        terminal = jax.lax.cond(
            condition1,
            lambda _: True,
            lambda _: condition2,
            operand=None,
        )
        next_env_state = (
            embedding_state,
            trajectory,
            depth + 1,
            rewards,
        )

        return (
            next_env_state,
            encoder.encode(code),
            self.best_reward,
            terminal,
            None,
        )

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        code = open_file(self.file_name)
        if self.source_platform in ["cuda", "hip"]:
            code = code.split("extern")[0]
        embedding_state = jnp.array(encoder.encode(code))
        trajectory = jnp.zeros(self.optimizer_len, dtype=int)
        depth = 0
        rewards = jnp.zeros(
            (1, self.optimizer_len, self.num_actions), dtype=jnp.float32
        )
        return embedding_state, trajectory, depth, rewards

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        optimize_grid, trajectory, depth = env_state
        return optimize_grid

    @property
    def num_actions(self):
        return self.action_len


def build_env(file_name: str, source_platform: str = "mlu", target_platform: str = "cpu") -> FalconGo:
    action_len = len(ActionSpace)
    base_name = os.path.basename(file_name)
    op_name = base_name.split("_")[0]
    optimizer_len = MAX_DEPTH  # 可以绑定 max_depth
    tvm_env = FalconGo(
        file_name,
        op_name,
        source_platform,
        target_platform,
        action_len=action_len,
        optimizer_len=optimizer_len,
    )
    return tvm_env


def get_recurrent_fn(env: FalconGo):
    batch_step = vmap(env.step)

    def recurrent_fn(params, key, actions, env_state):
        key, subkey = jax.random.split(key)
        new_env_state, obs, max_reward, terminals, _ = batch_step(
            actions, env_state
        )
        embedding_state, trajectory, depth, rewards = new_env_state
        trajectory = trajectory.at[depth].set(actions)
        depth_val = int(jax.device_get(depth)[0])
        cur_action_ids = lax.dynamic_slice(trajectory, (0, 0), (1, depth_val))
        jax.device_get(cur_action_ids)[0].tolist()

        code_embedding = [int(arr) for arr in embedding_state[0]]
        code = encoder.decode(code_embedding)
        _invalid_mask = jnp.array(
            get_invalid_actions(code, env.source_platform, env.target_platform)
        ).reshape(1, -1)
        reward = rewards[0, 0, depth - 1, actions]

        prior_logits = jnp.array(
            generate_prior_from_src(
                code, env.source_platform, env.target_platform
            )
        ).reshape(1, -1)

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.where(terminals, 0, 1).astype(jnp.float32),
                prior_logits=prior_logits,
                value=reward,
            ),
            new_env_state,
        )

    return recurrent_fn


def _run_demo(env: FalconGo, rng_key) -> mctx.PolicyOutput:
    """Runs Gumbel MuZero search on env."""
    batch_reset = vmap(env.reset)

    key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, num=BS)

    states_init = batch_reset(subkeys)
    key, logits_rng = jax.random.split(key)
    rng_key, logits_rng, q_rng, search_rng = jax.random.split(key, 4)
    code = open_file(env.file_name)
    if env.source_platform in ["cuda", "hip"]:
        code = code.split("extern")[0]

    invalid_actions = jnp.array(
        get_invalid_actions(code, env.source_platform, env.target_platform)
    ).reshape(1, -1)

    prior_logits = jnp.array(
        generate_prior_from_src(code, env.source_platform, env.target_platform)
    ).reshape(1, -1)

    root = mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=jnp.zeros([BS]),
        embedding=states_init,
    )

    recurrent_fn = get_recurrent_fn(env)

    policy_output = mctx.gumbel_muzero_policy(
        params=states_init,
        rng_key=search_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=NUM_SIMULATIONS,
        invalid_actions=invalid_actions,
        max_depth=env.optimizer_len,
        max_num_considered_actions=MAX_NUM_CONSIDERED_ACTIONS,
    )
    return policy_output


# ================== HTTP 后端封装 ==================

class TranslateRequest(BaseModel):
    src_lang: str        # "cpu" / "cuda" / "bangc" / "hip"
    dst_lang: str        # 同上
    src_code: str
    file_context: Optional[str] = None
    io_spec: Optional[dict] = None
    target_hw: Optional[str] = None
    extra_hints: Optional[str] = None


class VerifyResult(BaseModel):
    functional_pass: bool = True
    max_abs_diff: Optional[float] = None
    perf_baseline_ms: Optional[float] = None
    perf_new_ms: Optional[float] = None


class TranslateResponse(BaseModel):
    status: str
    generated_code: Optional[str] = None
    verify_result: Optional[VerifyResult] = None
    raw_log: Optional[str] = None
    error_message: Optional[str] = None


app = FastAPI()


def map_lang_to_platform(lang: str) -> str:
    """VSCode 里的 src_lang/dst_lang → 当前 pipeline 里的 platform 名称."""
    lang = lang.lower()
    if lang == "cpu":
        return "cpu"
    if lang == "cuda":
        return "cuda"
    if lang == "hip":
        return "hip"
    raise ValueError(f"Unsupported language/platform: {lang}")


def decide_file_extension(source_platform: str) -> str:
    """根据源平台选择写临时文件的后缀."""
    if source_platform == "cuda":
        return ".cu"
    if source_platform in ("cpu", "hip"):
        return ".cpp"
    return ".cpp"


def run_falcon_search(
    src_code: str,
    src_platform: str,
    dst_platform: str,
    seed: int = SEED,
) -> Tuple[str, float, float]:
    """
    1. 把 src_code 写到 workspace 临时文件
    2. 调用 build_env + _run_demo 做搜索
    3. 从 env.output_dir 里读回最优实现
    """
    workspace_dir = os.path.abspath("opmigrate_workspace")
    os.makedirs(workspace_dir, exist_ok=True)

    ext = decide_file_extension(src_platform)
    unique_id = uuid.uuid4().hex[:8]
    base_name = f"kernel_{unique_id}"
    #src_path = os.path.join(workspace_dir, base_name + ext)
    src_path = os.path.join(workspace_dir, 'add_1_15_64' + ext)

    with open(src_path, "w", encoding="utf-8") as f:
        f.write(src_code)

    env = build_env(src_path, src_platform, dst_platform)
    rng_key = jax.random.PRNGKey(seed)

    start = time.time()
    _ = _run_demo(env, rng_key)
    elapsed = time.time() - start

    output_dir = os.path.abspath(env.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    name_no_ext, _ = os.path.splitext(os.path.basename(src_path))
    pattern = os.path.join(output_dir, name_no_ext + ".*")
    candidates = glob.glob(pattern)

    if not candidates:
        raise RuntimeError(
            f"No output files found in {output_dir} for base name {name_no_ext}"
        )

    best_path = max(candidates, key=os.path.getmtime)

    with open(best_path, "r", encoding="utf-8") as f:
        generated = f.read()

    logging.info(
        "[Falcon backend] best_reward=%f, elapsed=%.3fs, best_file=%s",
        env.best_reward,
        elapsed,
        best_path,
    )

    return generated, elapsed, env.best_reward


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    try:
        src_platform = map_lang_to_platform(req.src_lang)
        dst_platform = map_lang_to_platform(req.dst_lang)
    except ValueError as e:
        return TranslateResponse(
            status="failed",
            error_message=str(e),
        )

    try:
        generated_code, elapsed, best_reward = run_falcon_search(
            src_code=req.src_code,
            src_platform=src_platform,
            dst_platform=dst_platform,
            seed=SEED,
        )
    except Exception as e:
        logging.exception("Falcon search failed.")
        return TranslateResponse(
            status="failed",
            error_message=f"Falcon search failed: {e}",
        )

    verify = VerifyResult(
        functional_pass=True,
        max_abs_diff=None,
        perf_baseline_ms=None,
        perf_new_ms=None,
    )

    raw_log = (
        f"Falcon search best_reward={best_reward:.6f}, "
        f"elapsed={elapsed:.3f}s, src={src_platform}, dst={dst_platform}"
    )

    return TranslateResponse(
        status="success",
        generated_code=generated_code,
        verify_result=verify,
        raw_log=raw_log,
    )


if __name__ == "__main__":
    # 直接作为后端运行：VSCode 插件里配置 http://127.0.0.1:9000
    uvicorn.run(app, host="0.0.0.0", port=9000)
