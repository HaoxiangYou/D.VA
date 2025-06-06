import gym
import numpy as np


class DeepMindControl:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        elif isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
    
class DeepMindControlDflex:
    metadata = {}

    def __init__(self, env, action_repeat=1, size=(64, 64), camera=None, seed=0):
        self._env = env
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = 0
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action, enable_reset = False, enable_vis_obs = True):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(actions=action, 
                                        enable_reset = enable_reset, 
                                        enable_vis_obs = enable_vis_obs)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = {} 
        obs["image"] = time_step.observation 
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        # Normalize Reward to compare with other baselines
        return obs, reward / self._action_repeat, done, info

    def reset(self, env_ids = None, force_reset = True, enable_vis_obs=True):
        time_step = self._env.reset(env_ids=env_ids, 
                                    force_reset=force_reset,
                                    enable_vis_obs=enable_vis_obs)

        obs = {}
        obs["image"] = time_step.observation
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs