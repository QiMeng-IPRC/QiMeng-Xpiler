import datetime
import logging
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import pgx
import pgx.core as core
import tiktoken
from jax import jit
from jax.experimental import io_callback
from network import CodeAZNet
from omegaconf import OmegaConf
from pgx._src.struct import dataclass
from pgx._src.types import Array
from pydantic import BaseModel

from benchmark.perf import perf_cuda, perf_dlboost, perf_hip, perf_mlu
from falcon.mcts.actions import actions as ActionSpace
from falcon.mcts.utils import open_file
from falcon.util import get_target

GFLOPS = 64 * 1280 * 2 / 1e9

encoder = tiktoken.encoding_for_model("gpt-4o")
devices = jax.local_devices()
num_devices = len(devices)

logging.basicConfig(
    level=logging.INFO,  # Set the log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Set the log format
)


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
        return -10000.0


class Config(BaseModel):
    seed: int = 0
    max_num_iters: int = 13
    # network params
    max_sequence_length: int = 400
    # selfplay params
    selfplay_batch_size: int = 1
    num_simulations: int = 400
    # Tree depth
    max_num_steps: int = 10
    # training params
    training_batch_size: int = 1
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5
    action_num: int = 11

    class Config:
        extra = "forbid"


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
logging.info(config)


@dataclass
class State(core.State):
    current_player: Array = jnp.array(0, dtype=jnp.int32)
    rewards: jax.Array = jnp.array(
        [0.0], dtype=jnp.float32
    )  # shape = (B, num_players)
    terminated: Array = jnp.array(False, dtype=jnp.bool_)
    truncated: Array = jnp.array(False, dtype=jnp.bool_)
    legal_action_mask: Array = jnp.ones(
        config.action_num, dtype=jnp.bool_
    )  # Enter your action count here.
    observation: Array = jnp.zeros(
        (150,), dtype=jnp.int32
    )  # Use the shape you actually coded.
    iteration: Array = jnp.array(0, dtype=jnp.int32)
    best_reward: Array = jnp.array(-10000.0, dtype=jnp.float32)
    _step_count: Array = jnp.array(0, dtype=jnp.int32)

    @property
    def env_id(self) -> core.EnvId:
        return "Falcon"  # type: ignore


class FalconGo:
    def __init__(
        self,
        file_name,
        op_name,
        source_platform,
        target_platform,
        action_len=config.action_num,
        optimizer_len=config.action_num,
    ):
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
            score = 0
        return code, np.float32(score)

    def perform_action_py(self, action_ids: jnp.ndarray) -> float:
        # This is a standard Python function; the action ID must be converted
        # using numpy's tolist() method.
        action_list = [ActionSpace[action_ids]]
        code, reward = self.perform_action(action_list)
        # encoding + pad /truncate to fix length
        encoded = encoder.encode(code)
        max_len = config.max_sequence_length
        if len(encoded) < max_len:
            padded = encoded + [0] * (max_len - len(encoded))  # pad
        else:
            padded = encoded[:max_len]  # truncate
        code_embedding = np.array(padded, dtype=np.int32)  # shape: [max_len]
        return reward, code_embedding

    def step(self, state: State, action_ids: jnp.ndarray) -> State:
        code_embedding_shape = (config.max_sequence_length,)
        # Use pure_callback for asynchronous calls to pure Python functions to
        # obtain rewards.
        reward, code_embedding = io_callback(
            self.perform_action_py,
            (
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct(code_embedding_shape, jnp.int32),
            ),
            action_ids,
        )
        terminated = (reward == -10000.0) | (
            state.iteration >= config.max_num_iters
        )

        best_reward = jnp.maximum(state.best_reward, reward)

        return State(
            observation=code_embedding,
            terminated=jnp.array(terminated),
            rewards=jnp.array([reward], dtype=jnp.float32),
            legal_action_mask=jnp.ones((config.action_num,), dtype=bool),
            current_player=jnp.array(0),
            iteration=state.iteration + 1,
            best_reward=best_reward,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> State:
        code = open_file(self.file_name)
        if self.source_platform in ["cuda", "hip"]:
            code = code.split("extern")[0]
        encoded = encoder.encode(code)
        max_len = config.max_sequence_length

        if len(encoded) < max_len:
            padded = encoded + [0] * (max_len - len(encoded))  # pad
        else:
            padded = encoded[:max_len]  # truncate

        return State(
            observation=np.array(padded, dtype=np.int32),
            legal_action_mask=jnp.ones(self.num_actions, dtype=jnp.bool_),
            rewards=jnp.array([0.0], dtype=jnp.float32),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            current_player=jnp.array(0),
            iteration=jnp.array(0),
            best_reward=jnp.array(-10000.0, dtype=jnp.float32),
            _step_count=jnp.array(0),
        )

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        optimize_grid, trajectory, depth = env_state
        return optimize_grid

    @property
    def num_actions(self):
        return self.action_len


def build_env(file_name, source_platform="cpu", target_platform="cuda"):
    action_len = config.action_num
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


env = build_env("benchmark/data/cpp_code_test/bmm_4_128_128_128.cpp")


def forward_fn(token_ids, is_eval=False):
    net = CodeAZNet()

    policy_out, value_output = net(
        token_ids, is_training=not is_eval, test_local_stats=False
    )
    return policy_out, value_output


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(
    model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State
):
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True
    )
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(
        state.legal_action_mask, logits, jnp.finfo(logits.dtype).min
    )
    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        root = mctx.RootFnOutput(
            prior_logits=logits, value=value, embedding=state
        )

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        jax.random.split(key2, batch_size)
        # state = jax.vmap(auto_reset(env.step, env.reset))(
        #     state, policy_output.action
        # )
        state = jax.vmap(env.step)(state, policy_output.action)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.reset)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    # mask if the episode is truncated
    value_loss = jnp.mean(value_loss * samples.mask)

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(
        loss_fn, has_aux=True
    )(model_params, model_state, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss


@jax.pmap
def evaluate(rng_key, model):
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    # Initialize the environmental state.
    batch_reset = jax.vmap(env.reset)
    key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, num=1)
    states = batch_reset(subkeys)

    best_rewards = jnp.zeros((1, batch_size))  # Best reward for each episode

    def body_fn(val):
        keys, states, best_rewards = val
        # Use models to predict actions.
        (logits, _), _ = forward.apply(
            model_params, model_state, states.observation, is_eval=True
        )

        actions = jnp.argmax(logits, axis=-1)
        # Execute the action.
        next_states = jax.vmap(env.step)(states, actions)

        # Update the best rewards.
        new_best = jnp.maximum(best_rewards, next_states.rewards)
        return (keys, next_states, new_best)

    # Run the entire episode.
    _, final_states, best_rewards = jax.lax.while_loop(
        lambda x: ~x[1].terminated.all(), body_fn, (keys, states, best_rewards)
    )

    # Return to the final reward and the best reward.
    return final_states.rewards, best_rewards


if __name__ == "__main__":
    # Initialize model and opt_state
    rng_key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, num=1)
    global_best_reward = 0.0
    state_init = env.reset(subkeys[0])  # No need to use vmap.
    model = forward.init(
        jax.random.PRNGKey(0), state_init.observation[None, :]
    )
    opt_state = optimizer.init(params=model[0])
    # replicates to all devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"falcon_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        if iteration % config.eval_interval == 0:
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)

            # Obtain the ultimate reward and the best reward.
            final_R, best_R = evaluate(keys, model)

            # Collect the results from all devices.
            all_final_R = final_R.reshape(-1)
            all_best_R = best_R.reshape(-1)

            # Calculate evaluation metrics
            avg_reward = jnp.mean(all_final_R).item()
            max_reward = jnp.max(all_final_R).item()
            min_reward = jnp.min(all_final_R).item()
            best_reward = jnp.max(all_best_R).item()
            success_rate = (all_final_R > 0).mean().item()

            log.update(
                {
                    "eval/avg_reward": avg_reward,
                    "eval/max_reward": max_reward,
                    "eval/min_reward": min_reward,
                    "eval/best_reward": best_reward,
                }
            )

            # Save the best model.
            if best_reward > global_best_reward:
                global_best_reward = best_reward
                # Code for saving the model...

                # Store checkpoints
                model_0, opt_state_0 = jax.tree_util.tree_map(
                    lambda x: x[0], (model, opt_state)
                )
                with open(
                    os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb"
                ) as f:
                    dic = {
                        "config": config,
                        "rng_key": rng_key,
                        "model": jax.device_get(model_0),
                        "opt_state": jax.device_get(opt_state_0),
                        "iteration": iteration,
                        "frames": frames,
                        "hours": hours,
                        "pgx.__version__": pgx.__version__,
                        "env_id": env.id,
                        "env_version": env.version,
                    }
                    pickle.dump(dic, f)

        print(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        samples: Sample = compute_loss_input(data)

        # Shuffle samples and make minibatches
        # (#devices, batch, max_num_steps, ...)
        samples = jax.device_get(samples)
        frames += (
            samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        )
        samples = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[3:])), samples
        )
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]),
            samples,
        )

        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(
                lambda x: x[i], minibatches
            )
            model, opt_state, policy_loss, value_loss = train(
                model, opt_state, minibatch
            )
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )
