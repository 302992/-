"""Completed rewards must survive full resets and clear on the next episode."""

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.envs.wrappers import training as brax_training
from brax.training.logger import EpisodeMetricsLogger

from mujoco_playground._src import mjx_env

from .training_metrics import wrap_for_training


class _CountingEnv:
    """Small deterministic task: two reward units each step, reset command=0."""

    action_size = 1
    observation_size = 1

    @property
    def unwrapped(self):
        return self

    def reset(self, rng):
        del rng
        return mjx_env.State(
            data=jp.zeros(1),
            obs=jp.zeros(1),
            reward=jp.zeros(()),
            done=jp.zeros(()),
            metrics={"marker": jp.zeros(())},
            info={"command": jp.zeros(())},
        )

    def step(self, state, action):
        del action
        return state.replace(
            data=state.data + 1,
            obs=state.data + 1,
            reward=jp.asarray(2.0),
            done=jp.zeros(()),
            metrics={**state.metrics, "marker": jp.asarray(3.0)},
        )


@pytest.mark.parametrize("jit", (False, True))
def test_full_reset_preserves_terminal_diagnostics_and_starts_new_totals(jit):
    wrapped = wrap_for_training(_CountingEnv(), episode_length=2, action_repeat=1)
    reset = jax.jit(wrapped.reset) if jit else wrapped.reset
    step = jax.jit(wrapped.step) if jit else wrapped.step
    state = reset(jax.random.split(jax.random.PRNGKey(0), 2))
    state = state.replace(info={**state.info, "command": jp.ones(2) * 7.0})
    action = jp.zeros((2, 1))

    state = step(state, action)
    np.testing.assert_array_equal(state.info["episode_done"], 0.0)
    np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], 2.0)
    np.testing.assert_array_equal(state.info["command"], 7.0)

    state = step(state, action)
    np.testing.assert_array_equal(state.done, 1.0)
    np.testing.assert_array_equal(state.info["episode_done"], 1.0)
    np.testing.assert_array_equal(state.info["truncation"], 1.0)
    np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], 4.0)
    np.testing.assert_array_equal(state.info["episode_metrics"]["length"], 2.0)
    np.testing.assert_array_equal(state.info["episode_metrics"]["marker"], 6.0)
    # Physics and command belong to the fresh episode, not the completed one.
    np.testing.assert_array_equal(state.data, 0.0)
    np.testing.assert_array_equal(state.obs, 0.0)
    np.testing.assert_array_equal(state.info["command"], 0.0)

    state = step(state, action)
    np.testing.assert_array_equal(state.info["episode_done"], 0.0)
    np.testing.assert_array_equal(state.info["truncation"], 0.0)
    np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], 2.0)
    np.testing.assert_array_equal(state.info["episode_metrics"]["length"], 1.0)
    np.testing.assert_array_equal(state.info["episode_metrics"]["marker"], 3.0)
    np.testing.assert_array_equal(state.data, 1.0)


def test_eval_wrapper_still_observes_the_completed_episode_reward():
    wrapped = wrap_for_training(_CountingEnv(), episode_length=2, action_repeat=1)
    evaluated = brax_training.EvalWrapper(wrapped)
    state = jax.jit(evaluated.reset)(jax.random.split(jax.random.PRNGKey(1), 2))
    step = jax.jit(evaluated.step)
    action = jp.zeros((2, 1))
    state = step(state, action)
    state = step(state, action)
    np.testing.assert_array_equal(state.done, 1.0)
    np.testing.assert_array_equal(state.info["eval_metrics"].episode_metrics["reward"], 4.0)
    np.testing.assert_array_equal(state.info["eval_metrics"].episode_steps, 2.0)
    np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], 4.0)


def test_native_brax_logger_receives_completed_reward_instead_of_reset_zero():
    wrapped = wrap_for_training(_CountingEnv(), episode_length=2, action_repeat=1)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(2), 2))
    step = jax.jit(wrapped.step)
    emitted = []
    logger = EpisodeMetricsLogger(
        steps_between_logging=4,
        progress_fn=lambda num_steps, metrics: emitted.append((num_steps, metrics)),
    )
    for _ in range(2):
        state = step(state, jp.zeros((2, 1)))
        logger.update_episode_metrics(
            state.info["episode_metrics"], state.info["episode_done"], {}
        )
    assert len(emitted) == 1
    assert emitted[0][0] == 4
    assert emitted[0][1]["episode/sum_reward"] == 4.0
    assert emitted[0][1]["episode/length"] == 2.0
