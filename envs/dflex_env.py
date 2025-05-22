# License: see [LICENSE, LICENSES/DiffRL/LICENSE]

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

import dflex as df
import xml.etree.ElementTree as ET

from gym import spaces
from utils.copy_utils import safe_deepcopy
from viewer.maniskill_viewer import ManiskillViewer


class DFlexEnv:
    
    def __init__(self, num_envs, state_num_obs, num_act, episode_length, MM_caching_frequency = 1, seed=0, no_grad=True, device='cuda:0', img_height=84, img_width=84):
        self.seed = seed

        self.no_grad = no_grad
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
            (num_envs, 9, self.obs_img_height, self.obs_img_width), device=self.device, dtype=torch.uint8, requires_grad=False)
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

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def state_observation_space(self):
        return self.state_obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_state_obs(self):
        return self.state_num_observations
    
    @property
    def num_vis_obs(self):
        return (9, self.obs_img_height, self.obs_img_width) 

    def get_state(self):
        return self.state.joint_q.clone().view(self.num_envs, -1), self.state.joint_qd.clone().view(self.num_envs, -1)

    def reset_with_state(self, init_joint_q, init_joint_qd, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # fixed start state
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, :] = init_joint_q.view(-1, self.num_joint_q)[env_ids, :].clone()
            self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = init_joint_qd.view(-1, self.num_joint_qd)[env_ids, :].clone()
            
            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.state_obs_buf
    
    def clone(self):
        # Create a new, blank instance of the class
        new_env = self.__class__.__new__(self.__class__)
        
        # Iterate through all attributes and handle them
        for k, v in self.__dict__.items():
            if isinstance(v, ManiskillViewer):
                # If it's a Viewer, create a reference instead of new instance 
                setattr(new_env, k, v)
            else:
                setattr(new_env, k, safe_deepcopy(v))  # Recursively deepcopy other attributes
                
        return new_env
