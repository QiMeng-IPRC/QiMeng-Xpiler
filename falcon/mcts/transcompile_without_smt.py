import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import mctx
import tiktoken
from absl import app, flags
from jax import jit, lax, vmap

from benchmark.perf import perf_cuda, perf_dlboost, perf_hip, perf_mlu
from falcon.mcts.action_logit import generate_prior_from_src
from falcon.mcts.action_without_smt import actions as ActionSpace
from falcon.mcts.invalid_actions import get_invalid_actions
from falcon.mcts.utils import open_file
from falcon.util import get_target

# Configure the log
logging.basicConfig(
    level=logging.INFO,  # Set the log level.
    format="%(asctime)s - %(levelname)s - %(message)s",  # Set the log format.
)

# TODO(michale): replace with shape calculation
GFLOPS = 64 * 1280 * 2 / 1e9
A_Length = len(ActionSpace)


encoder = tiktoken.encoding_for_model("gpt-4o")
FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_simulations", 512, "Number of simulations.")
flags.DEFINE_integer(
    "max_num_considered_actions",
    13,
    "The maximum number of actions expanded at the root.",
)
flags.DEFINE_integer("max_depth", 13, "The maximum search depth.")
flags.DEFINE_string("source", "mlu", "Source platform identifier.")
flags.DEFINE_string("target", "cpu", "Destination platform identifier.")
flags.DEFINE_string(
    "file_name",
    "benchmark/data/mlu_code_test/add_3_3_256.mlu",
    "Path to the input kernel file.",
)
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
    except Exception:
        logging.info(e)
        return 0.0


class FalconGo:
    def __init__(
        self,
        file_name,
        op_name,
        source_platform,
        target_platform,
        action_len=A_Length,
        optimizer_len=A_Length,
        goal_reward=False,
        timeout=None,
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
        # Ensure the directory exists.
        os.makedirs(self.output_dir, exist_ok=True)

    def perform_action(self, actions):
        """Generates a design space for a given `action`.

        It calls `generate_design_space()`
        with specific parameters to apply the given scheduling rule (`action`) to the module.
        The function returns a new `ProgramState` object, which represents the new program
        state after applying the action.
        """
        code = open_file(self.file_name)
        code = (
            code.split("extern")[0]
            if self.source_platform in ["cuda", "hip"]
            else code
        )
        for action in actions:
            code = action(
                self.file_name,
                code,
                self.source_platform,
                self.target_platform,
            )

        target, file_type = get_target(code, self.target_platform)
        os.makedirs("tmp", exist_ok=True)
        # Extract base name and replace extension
        base_name = os.path.basename(self.file_name)
        name_no_ext, _ = os.path.splitext(base_name)
        new_file = os.path.join("tmp", name_no_ext + file_type)
        with open(new_file, "w", encoding="utf-8") as f:
            f.write(code)
        score = objective(new_file, target)
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
                # save the file into success transcompiled folder
                # Extract base name and replace extension
                base_name = os.path.basename(self.file_name)
                name_no_ext, _ = os.path.splitext(base_name)
                target, file_type = get_target(code, self.target_platform)
                new_file = os.path.join(
                    self.output_dir, name_no_ext + file_type
                )
                with open(new_file, "w", encoding="utf-8") as f:
                    f.write(code)
        except Exception:
            code = ""
            reward = -10000
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
            new_value = (rewards[0, depth_index, var] + reward) / 2
            rewards = rewards.at[0, depth_index, var].set(new_value)

        # Treminated if we reach the goal or the reward is zero
        condition1 = depth > self.optimizer_len
        condition2 = reward == -10000

        terminal = jax.lax.cond(
            condition1,
            lambda _: True,  # If condition1 is True
            lambda _: condition2,  # If condition1 is False, return condition2
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
        embedding_state = jnp.array(encoder.encode(open_file(self.file_name)))
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


def build_env(file_name, source_platform="mlu", target_platform="cpu"):
    action_len = len(ActionSpace)
    base_name = os.path.basename(file_name)
    op_name = base_name.split("_")[0]
    optimizer_len = 14
    tvm_env = FalconGo(
        file_name,
        op_name,
        source_platform,
        target_platform,
        action_len=action_len,
        optimizer_len=optimizer_len,
    )
    return tvm_env


def get_recurrent_fn(env):
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
        invalid_mask = jnp.array(
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


def _run_demo(env, rng_key):
    """Runs a search algorithm on a toy environment."""
    batch_reset = vmap(env.reset)

    key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, num=BS)

    states_init = batch_reset(subkeys)
    key, logits_rng = jax.random.split(key)
    rng_key, logits_rng, q_rng, search_rng = jax.random.split(key, 4)
    code = open_file(env.file_name)
    code = (
        code.split("extern")[0]
        if env.source_platform in ["cuda", "hip"]
        else code
    )
    invalid_actions = jnp.array(
        get_invalid_actions(code, env.source_platform, env.target_platform)
    ).reshape(1, -1)
    # prior_logits = jnp.ones((1, A_Length)) / A_Length
    prior_logits = jnp.array(
        generate_prior_from_src(code, env.source_platform, env.target_platform)
    ).reshape(1, -1)
    root = mctx.RootFnOutput(
        prior_logits=prior_logits,  # jnp.full([batch_size, num_actions],
        value=jnp.zeros([BS]),
        # The embedding will hold the state index.
        embedding=states_init,
    )

    recurrent_fn = get_recurrent_fn(env)
    # Running the search.
    policy_output = mctx.gumbel_muzero_policy(
        params=states_init,
        rng_key=search_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=FLAGS.num_simulations,
        invalid_actions=invalid_actions,
        max_depth=env.optimizer_len,
        max_num_considered_actions=FLAGS.max_num_considered_actions,
    )
    return policy_output


def main(argv):

    rng_key = jax.random.PRNGKey(FLAGS.seed)
    falcon_env = build_env(FLAGS.file_name, FLAGS.source, FLAGS.target)

    start_time = time.time()
    policy_output = _run_demo(falcon_env, rng_key)
    batch_index = 0
    selected_action = policy_output.action[batch_index]
    q_value = policy_output.search_tree.summary().qvalues[
        batch_index, selected_action
    ]
    # To estimate the value of the root state, use the Q-value of the selected
    # action. The Q-value is not affected by the exploration at the root
    # node.
    end_time = time.time()
    print(f"[INFO]searching time: {end_time - start_time} s")


if __name__ == "__main__":
    app.run(main)
