import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import numpy as np
import torch
import argparse
import os
import math
import sys
import random
import time
import json
import copy
import yaml
import envs
from viewer.dmc_viewer import DMCViewer
import externals.curl.utils as utils
from externals.curl.logger import Logger
from externals.curl.video import VideoRecorder

from externals.curl.curl_sac import CurlSacAgent
from torchvision import transforms
import dm_env
from gymnasium import core, spaces
from train_curl import load_env


with open("./cfg/curl/hopper.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

with open("./cfg/curl/hopper_test.yaml", 'r') as f:
    cfg_test = yaml.load(f, Loader=yaml.SafeLoader)

train_env = load_env(cfg, eval=False)
test_env = load_env(cfg_test, eval=False)
obs_test = test_env.reset(env_ids = None, force_reset = True, enable_vis_obs=True)
obs_train = train_env.reset(env_ids = None, force_reset = True, enable_vis_obs=True)


# make sure the initialization is the same
assert np.array_equal(obs_test, obs_train)

# use the same action for testing action repeat process 
action = train_env.action_space.sample()

# use 25 steps for train envs with action repeat 4
reward_train_cumulative = np.zeros(cfg["params"]["general"]["num_actors"])
steps = 10
for _ in range(steps):
    next_obs_train, reward_train, done_train, _ = train_env.step(action, enable_reset = False, enable_vis_obs = True)
    done_env_ids_train = done_train.nonzero()[0]
    if done_env_ids_train.size != 0:
        print(f"reset training envs: {done_env_ids_train}")
        train_env.reset(env_ids=np.array(done_env_ids_train, dtype=np.int32), force_reset = False, enable_vis_obs=True)
    reward_train_cumulative += reward_train

# use 100 steps for test envs with action_repeat 1
reward_test_cumulative = np.zeros(cfg_test["params"]["general"]["num_actors"])
for _ in range(4 * steps):
    next_obs_test, reward_test, done_test, _ = test_env.step(action, enable_reset = False, enable_vis_obs = True)
    done_env_ids_test = done_test.nonzero()[0]
    if done_env_ids_test.size != 0:
        print(f"reset testing envs: {done_env_ids_test}")
        test_env.reset(env_ids=np.array(done_env_ids_test, dtype=np.int32), force_reset = False, enable_vis_obs=True)
    reward_test_cumulative += reward_test

assert np.array_equal(next_obs_test, next_obs_train)
print("pass observation test")

assert np.array_equal(done_env_ids_train, done_env_ids_test)
print("pass termination test")

assert np.linalg.norm(reward_train_cumulative-reward_test_cumulative) <= 1e-5
print("pass reward test")
