# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY = (
    "`env.action_spec()` must return a single `BoundedArray`, got: {}.")
_MUST_BE_FINITE = "All values in `{name}` must be finite, got: {bounds}."
_MUST_BROADCAST = (
    "`{name}` must be broadcastable to shape {shape}, got: {bounds}.")


class ActionScaleWrapper(dm_env.Environment):
  """Wraps a control environment to rescale actions to a specific range."""
  __slots__ = ("_action_spec", "_env", "_transform")

  def __init__(self, env, minimum, maximum):
    """Initializes a new action scale Wrapper.

    Args:
      env: Instance of `dm_env.Environment` to wrap. Its `action_spec` must
        consist of a single `BoundedArray` with all-finite bounds.
      minimum: Scalar or array-like specifying element-wise lower bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.
      maximum: Scalar or array-like specifying element-wise upper bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.

    Raises:
      ValueError: If `env.action_spec()` is not a single `BoundedArray`.
      ValueError: If `env.action_spec()` has non-finite bounds.
      ValueError: If `minimum` or `maximum` contain non-finite values.
      ValueError: If `minimum` or `maximum` are not broadcastable to
        `env.action_spec().shape`.
    """
    action_spec = env.action_spec()
    if not isinstance(action_spec, specs.BoundedArray):
      raise ValueError(_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec))

    minimum = np.array(minimum)
    maximum = np.array(maximum)
    shape = action_spec.shape
    orig_minimum = action_spec.minimum
    orig_maximum = action_spec.maximum
    orig_dtype = action_spec.dtype

    def validate(bounds, name):
      if not np.all(np.isfinite(bounds)):
        raise ValueError(_MUST_BE_FINITE.format(name=name, bounds=bounds))
      try:
        np.broadcast_to(bounds, shape)
      except ValueError:
        raise ValueError(_MUST_BROADCAST.format(
            name=name, bounds=bounds, shape=shape))

    validate(minimum, "minimum")
    validate(maximum, "maximum")
    validate(orig_minimum, "env.action_spec().minimum")
    validate(orig_maximum, "env.action_spec().maximum")

    scale = (orig_maximum - orig_minimum) / (maximum - minimum)

    def transform(action):
      new_action = orig_minimum + scale * (action - minimum)
      return new_action.astype(orig_dtype, copy=False)

    dtype = np.result_type(minimum, maximum, orig_dtype)
    self._action_spec = action_spec.replace(
        minimum=minimum, maximum=maximum, dtype=dtype)
    self._env = env
    self._transform = transform

  def step(self, actions, enable_reset = False, enable_vis_obs = True):
    return self._env.step(actions = self._transform(actions), 
                            enable_reset = enable_reset, 
                            enable_vis_obs = enable_vis_obs)

  def reset(self, env_ids = None, force_reset = True, enable_vis_obs=True):
    return self._env.reset(env_ids=env_ids, 
                           force_reset=force_reset,
                           enable_vis_obs=enable_vis_obs)

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._action_spec

  def __getattr__(self, name):
    return getattr(self._env, name)

class ActionRepeatMultiEnvsWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, actions, enable_reset = False, enable_vis_obs = True):
        reward = np.zeros(self._env.num_envs, np.float32)
        discount = np.ones(self._env.num_envs, np.float32)
        done_envs = {} # this keep track of which envs are done so that we don't repeat them  
        for i in range(self._num_repeats):
            time_steps = self._env.step(actions = actions, 
                                        enable_reset = enable_reset, 
                                        enable_vis_obs = enable_vis_obs)
            for j, time_step in enumerate(time_steps):
                if j in done_envs.keys():
                    # Discard the current time_step if the env has reached the end previously
                    time_steps[j] = done_envs[j]
                elif time_step.last():
                    # if the env is done, we will add the time_step to the done_envs
                    done_envs[j] = time_step
                    reward[j] += (time_step.reward or 0.0) * discount[j]
                else:
                    # if the envs is not done, we will add the reward and discount
                    reward[j] += (time_step.reward or 0.0) * discount[j]
                    discount[j] *= time_step.discount
        # store done envs idx for reset purpose
        self.done_envs = np.array(list(done_envs.keys()), dtype=np.int32)
        return [time_step._replace(reward=reward[idx], discount=discount[idx]) for idx, time_step in enumerate(time_steps)]

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self, env_ids = None, force_reset = True, enable_vis_obs=True):
        return self._env.reset(env_ids=env_ids, 
                                force_reset=force_reset,
                                enable_vis_obs=enable_vis_obs)

    def __getattr__(self, name):
        return getattr(self._env, name)
    
class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, actions, enable_reset = False, enable_vis_obs = True):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(actions=actions, 
                                        enable_reset = enable_reset, 
                                        enable_vis_obs = enable_vis_obs)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self, env_ids = None, force_reset = True, enable_vis_obs=True):
        return self._env.reset(env_ids=env_ids, 
                               force_reset=force_reset,
                               enable_vis_obs=enable_vis_obs)

    def __getattr__(self, name):
        return getattr(self._env, name)

class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, actions):
        time_step = self._env.step(actions)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, actions, enable_reset = False, enable_vis_obs = True):
        actions = actions.astype(self._env.action_spec().dtype)
        return self._env.step(actions=actions, 
                                enable_reset = enable_reset, 
                                enable_vis_obs = enable_vis_obs)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self, env_ids = None, force_reset = True, enable_vis_obs=True):
        return self._env.reset(env_ids=env_ids, 
                               force_reset=force_reset,
                               enable_vis_obs=enable_vis_obs)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, env_ids = None, force_reset = True, enable_vis_obs=True):
        time_step = self._env.reset(env_ids=env_ids, 
                                    force_reset=force_reset,
                                    enable_vis_obs=enable_vis_obs)
        return self._augment_time_step(time_step)

    def step(self, actions, enable_reset = False, enable_vis_obs = True):
        time_step = self._env.step(actions=actions, 
                                   enable_reset = enable_reset, 
                                   enable_vis_obs = enable_vis_obs)
        return self._augment_time_step(time_step, actions)

    def _augment_time_step(self, time_step, actions=None):
        if actions is None:
            action_spec = self.action_spec()
            actions = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=actions,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def make(name, frame_stack, action_repeat, seed):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env


class ExtendedTimeStepMultiEnvsWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, env_ids = None, force_reset = True, enable_vis_obs=True):
        time_step = self._env.reset(env_ids=env_ids, 
                                    force_reset=force_reset,
                                    enable_vis_obs=enable_vis_obs)
        return self._augment_time_step(time_step)

    def step(self, actions, enable_reset = False, enable_vis_obs = True):
        time_step = self._env.step(actions=actions, 
                                   enable_reset = enable_reset, 
                                   enable_vis_obs = enable_vis_obs)
        return self._augment_time_step(time_step, actions)

    def _augment_time_step(self, time_step_list, actions=None):
        if actions is None:
            action_spec = self.action_spec()
            actions = np.zeros((self._env.num_envs, self._env.num_actions), dtype=action_spec.dtype)
        return [ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=actions[idx],
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0) for idx, time_step in enumerate(time_step_list)]

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def make_env(env, cfg):
    action_repeat = cfg.action_repeat 
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatMultiEnvsWrapper(env, action_repeat)
    env = ActionScaleWrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepMultiEnvsWrapper(env)
    return env
