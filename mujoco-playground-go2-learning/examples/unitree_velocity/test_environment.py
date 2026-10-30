"""Behaviour checks for the teaching task; run with JAX_PLATFORMS=cpu."""

import hashlib

import jax
import jax.numpy as jp
import numpy as np
import pytest

from mujoco_playground._src import mjx_env

from .environment import command_vector, make_env
from .run import training_wrapper


@pytest.fixture(scope="module", params=("go2", "go1"))
def task(request):
    robot = request.param
    model_dir = mjx_env.MENAGERIE_PATH / f"unitree_{robot}"
    if not (model_dir / "assets").is_dir():
        pytest.skip(f"Local {robot} assets are not installed: {model_dir}")
    if robot == "go2" and not (model_dir / "go2_mjx.xml").is_file():
        pytest.skip("Local Go2 MJX model is not installed")
    env = make_env(robot, config_overrides={"noise_config.level": 0.0})
    return env, jax.jit(env.reset), jax.jit(env.step)


def test_manual_commands_survive_expired_timer_and_reach_policy_observation(task):
    """Policy inputs must represent all requested directions on every step."""
    env, reset, step = task
    set_command = jax.jit(env.set_command)
    expected_commands = {
        "stop": [0.0, 0.0, 0.0],
        "forward": [0.5, 0.0, 0.0],
        "backward": [-0.5, 0.0, 0.0],
        "left": [0.0, 0.0, 0.6],
        "right": [0.0, 0.0, -0.6],
    }
    for seed, (name, expected) in enumerate(expected_commands.items()):
        state = reset(jax.random.PRNGKey(seed))
        assert state.obs["state"].shape == (48,)
        assert state.obs["privileged_state"].shape == (123,)
        assert env.action_size == 12
        state = set_command(state, command_vector(name))
        np.testing.assert_allclose(state.obs["state"][-3:], expected)
        for _ in range(3):
            # Simulate a due/expired sampling timer.  An externally requested
            # command must still win over the training task's random sampler.
            info = dict(state.info)
            info["steps_until_next_cmd"] = jp.int32(0)
            state = step(state.replace(info=info), jp.zeros(env.action_size))
            np.testing.assert_allclose(state.info["command"], expected)
            np.testing.assert_allclose(state.obs["state"][-3:], expected)
            assert np.isfinite(np.asarray(state.data.qpos)).all()
            assert np.isfinite(float(state.reward))


def test_full_reset_replaces_manual_info_and_keeps_new_command_visible(task):
    """An episode boundary must reset both physics and command/history info."""
    env, _, _ = task
    wrapped = training_wrapper(env, episode_length=2, action_repeat=1)
    reset = jax.jit(wrapped.reset)
    step = jax.jit(wrapped.step)
    state = reset(jax.random.split(jax.random.PRNGKey(42), 2))
    # This distinctive command cannot come from the training command table.
    manual = jp.asarray([1.23, -0.72, 0.91], dtype=jp.float32)
    state = jax.jit(jax.vmap(env.set_command, in_axes=(0, None)))(state, manual)
    action = jp.zeros((2, env.action_size))
    state = step(state, action)
    assert np.asarray(state.info["manual_command"]).all()
    state = step(state, action)
    assert np.asarray(state.done).all()
    assert not np.asarray(state.info["manual_command"]).any()
    np.testing.assert_allclose(
        state.obs["state"][..., -3:], state.info["command"]
    )
    assert not np.allclose(np.asarray(state.info["command"]), np.asarray(manual))


def test_go2_adapter_leaves_external_xml_unchanged():
    """Adapting a local model must not overwrite downloaded source assets."""
    model_dir = mjx_env.MENAGERIE_PATH / "unitree_go2"
    paths = [model_dir / name for name in ("go2_mjx.xml", "scene_mjx.xml")]
    if not all(path.is_file() for path in paths):
        pytest.skip("Local Go2 MJX model is not installed")
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    env = make_env("go2", model_dir=model_dir)
    assert env.mj_model.nu == 12
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before
