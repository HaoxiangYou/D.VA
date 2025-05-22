from envs.dflex_env import DFlexEnv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df
import torch
import numpy as np
from gym import spaces


class DFlexDiffRenderEnv(DFlexEnv):

    def __init__(self, num_envs, state_num_obs, num_act, episode_length, MM_caching_frequency = 1, seed=0, no_grad=True, no_vis_grad=True, device='cuda:0', img_height=84, img_width=84):

        self.seed = seed

        self.no_grad = no_grad
        self.no_vis_grad = no_grad | no_vis_grad
        df.config.no_grad = self.no_grad

        self.episode_length = episode_length

        self.device = device

        self.sim_time = 0.0

        # visual obs related variables
        self.obs_img_height = img_height
        self.obs_img_width = img_width

        self.num_environments = num_envs
        self.num_agents = 1

        self.MM_caching_frequency = MM_caching_frequency
        
        # initialize observation and action space
        self.state_num_observations = state_num_obs
        self.num_actions = num_act

        self.state_obs_space = spaces.Box(np.ones(self.state_num_observations) * -np.Inf, np.ones(self.state_num_observations) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        # allocate buffers
        self.state_obs_buf = torch.zeros(
            (self.num_envs, self.state_num_observations), device=self.device, dtype=torch.float, requires_grad=False)
        # visual observation buffers
        self.vis_obs_buf = torch.zeros(
            (num_envs, 9, self.obs_img_height, self.obs_img_width), device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        # end of the episode
        self.termination_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device = self.device, dtype = torch.float, requires_grad = False)

        self.extras = {}