"""Keep completed-episode diagnostics visible after a full task reset.

Playground's full reset correctly replaces task history and commands, but also
replaces EpisodeWrapper's terminal diagnostics.  Brax's PPO logger reads these
fields from the returned state.  Cache just these fields through the existing
preserve-info hook; leave physics and all other task state fully reset.
"""

from mujoco_playground import wrapper

_PRESERVE_KEY = "AutoResetWrapper_preserve_info"
_CACHE_KEY = "unitree_velocity_episode_diagnostics"
_DIAGNOSTIC_FIELDS = ("episode_metrics", "episode_done", "truncation")


class _CacheEpisodeDiagnostics(wrapper.Wrapper):
    """Runs immediately after EpisodeWrapper and before the automatic reset."""

    @staticmethod
    def _cache(state):
        info = dict(state.info)
        preserved = dict(info.get(_PRESERVE_KEY, {}))
        preserved[_CACHE_KEY] = {name: info[name] for name in _DIAGNOSTIC_FIELDS}
        info[_PRESERVE_KEY] = preserved
        return state.replace(info=info)

    def reset(self, rng):
        return self._cache(self.env.reset(rng))

    def step(self, state, action):
        return self._cache(self.env.step(state, action))


class _ExposeEpisodeDiagnostics(wrapper.Wrapper):
    """Expose the completed transition, while keeping reset task state fresh."""

    def step(self, state, action):
        state = self.env.step(state, action)
        info = dict(state.info)
        info.update(info[_PRESERVE_KEY][_CACHE_KEY])
        return state.replace(info=info)


def wrap_for_training(env, **kwargs):
    """Use full task resets without losing terminal rewards or timeout flags.

    At an episode boundary the returned data/obs and task-specific info already
    belong to the next episode, while episode_metrics/episode_done/truncation
    describe the just-completed transition.  On the next step, EpisodeWrapper
    sees the previous episode_done flag and clears the old running totals.

    This local teaching wrapper supports the state-observation Go1/Go2 tasks.
    """
    wrapped = wrapper.wrap_for_brax_training(env, full_reset=True, **kwargs)
    if not isinstance(wrapped, wrapper.BraxAutoResetWrapper):
        raise TypeError("The Unitree training wrapper expects a state-observation task")
    # Only change the newly constructed wrapper chain, never the original
    # environment class or Playground's source/global definitions.
    wrapped.env = _CacheEpisodeDiagnostics(wrapped.env)
    return _ExposeEpisodeDiagnostics(wrapped)
