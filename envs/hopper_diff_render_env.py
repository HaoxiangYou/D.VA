
#from numpy.lib.function_base import angle
from envs.dflex_diff_render_env import DFlexDiffRenderEnv
import math
import torch

import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from copy import deepcopy

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

from utils import load_utils as lu
from utils import torch_utils as tu
from viewer.torch3d_robo import HopperRenderer

class HopperDiffRenderEnv(DFlexDiffRenderEnv):

    def __init__(self, device='cuda:0', num_envs=4096, seed=0, episode_length=1000, img_height=84, img_width=84, no_grad=True, 
                no_vis_grad=True, stochastic_init=False, MM_caching_frequency=1, early_termination=True):
        num_state_obs = 11
        num_act = 3
    
        super(HopperDiffRenderEnv, self).__init__(num_envs, num_state_obs, num_act, episode_length, MM_caching_frequency, seed, 
                                    no_grad=no_grad, no_vis_grad=no_vis_grad, device=device, img_height=img_height, img_width=img_width)

        self.stochastic_init = stochastic_init
        self.early_termination = early_termination

        self.init_sim()

        self.renderer = HopperRenderer(img_height=img_height, img_width=img_width, device=device)

        # other parameters
        self.termination_height = -0.45
        self.termination_angle = np.pi / 6.
        self.termination_height_tolerance = 0.15
        self.termination_angle_tolerance = 0.05
        self.height_rew_scale = 1.0
        self.action_strength = 200.0
        self.action_penalty = -1e-1

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0/60.0
        self.sim_substeps = 16
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 6
        self.num_joint_qd = 6

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.start_rotation = torch.tensor([0.], device = self.device, requires_grad = False)

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()

        self.start_pos = []
        self.start_joint_q = [0., 0., 0.]
        self.start_joint_target = [0., 0., 0.]

        start_height = 0.0

        asset_folder = os.path.join(os.path.dirname(__file__), 'assets')
        for i in range(self.num_environments):

            link_start = len(self.builder.joint_type)

            lu.parse_mjcf(os.path.join(asset_folder, "hopper.xml"), self.builder,
                density=1000.0,
                stiffness=0.0,
                damping=2.0,
                contact_ke=2.e+4,
                contact_kd=1.e+3,
                contact_kf=1.e+3,
                contact_mu=0.9,
                limit_ke=1.e+3,
                limit_kd=1.e+1,
                armature=1.0,
                radians=True, load_stiffness=True)

            self.builder.joint_X_pj[link_start] = df.transform((0.0, 0.0, 0.0), df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5))

            # base transform
            self.start_pos.append([0.0, start_height])

            # set joint targets to rest pose in mjcf
            self.builder.joint_q[i*self.num_joint_q + 3:i*self.num_joint_q + 6] = [0., 0., 0.]
            self.builder.joint_target[i*self.num_joint_q + 3:i*self.num_joint_q + 6] = [0., 0., 0., 0.]
        
        self.start_pos = tu.to_torch(self.start_pos, device=self.device)
        self.start_joint_q = tu.to_torch(self.start_joint_q, device=self.device)
        self.start_joint_target = tu.to_torch(self.start_joint_target, device=self.device)

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=self.device)

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()

        if (self.model.ground):
            self.model.collide(self.state)

    """
    This function render imgs for target envs
    """
    def render(self, env_ids):
        mujoco_joint_qs = self.get_mujoco_joint_q(self.state.joint_q.view(self.num_envs, -1))
        # scale such it has similar magnitude to uint8
        pixels = self.renderer.render(mujoco_joint_qs)[env_ids,:,:,:3] * 255
        return pixels

    """
    This function render a given trajectory (time sequences of joint_q in shac conventions) in shape (traj_length, num_env, num_q)
    """
    @torch.no_grad()
    def render_traj(self, traj, recording=False):
        frames = []
        camera_id = 1 if recording else 0
        for mujoco_joint_qs in traj:
            mujoco_joint_qs = self.get_mujoco_joint_q(mujoco_joint_qs)
            frames.append(
                (self.renderer.render(mujoco_joint_qs, camera_id=camera_id)[:,:,:,:3] * 255).cpu().to(torch.uint8)
            )
        return torch.stack(frames)
    
    def step(self, actions, enable_reset = True, enable_vis_obs = False):
        actions = actions.view((self.num_envs, self.num_actions))

        actions = torch.clip(actions, -1., 1.)

        self.actions = actions.clone()

        self.state.joint_act.view(self.num_envs, -1)[:, 3:] = actions * self.action_strength
        
        self.state = self.integrator.forward(self.model, self.state, self.sim_dt, self.sim_substeps, self.MM_caching_frequency)
        self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1

        self.calculateStateObservations()
        self.calculateReward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if enable_vis_obs:
            if enable_reset:
                if len(env_ids) < self.num_envs:
                    if self.no_vis_grad:
                        with torch.no_grad():
                            self.calculateVisualObservations((self.reset_buf == 0).nonzero(as_tuple=False).squeeze(-1))
                    else:
                        self.calculateVisualObservations((self.reset_buf == 0).nonzero(as_tuple=False).squeeze(-1))
            else:
                if self.no_vis_grad:
                    with torch.no_grad():
                        self.calculateVisualObservations(torch.arange(self.num_envs, dtype=torch.long, device=self.device))
                else:
                    self.calculateVisualObservations(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

        if self.no_grad == False and enable_reset == True:
            self.state_obs_buf_before_reset = self.state_obs_buf.clone()
            self.extras = {
                'state_obs_before_reset': self.state_obs_buf_before_reset,
                'episode_end': self.termination_buf
                }
            if enable_vis_obs:
                self.vis_obs_buf_before_reset = self.vis_obs_buf.clone()
                self.extras["vis_obs_before_reset"] = self.vis_obs_buf_before_reset
        
        if enable_reset:
            if len(env_ids) > 0:
                self.reset(env_ids, enable_vis_obs=enable_vis_obs)

        obs = {"state_obs": self.state_obs_buf}
        if enable_vis_obs:
            obs["vis_obs"] = self.vis_obs_buf

        return obs, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self, env_ids = None, force_reset = True, enable_vis_obs=False):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # clone the state to avoid gradient error
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()

            # fixed start state
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:2] = self.start_pos[env_ids, :].clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 2] = self.start_rotation.clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:] = self.start_joint_q.clone()
            self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.

            # randomization
            if self.stochastic_init:
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:2] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:2] + 0.05 * (torch.rand(size=(len(env_ids), 2), device=self.device) - 0.5) * 2.
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 2] = (torch.rand(len(env_ids), device = self.device) - 0.5) * 0.1
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:] + 0.05 * (torch.rand(size=(len(env_ids), self.num_joint_q - 3), device = self.device) - 0.5) * 2.
                self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.05 * (torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5) * 2.

            # clear action
            self.actions = self.actions.clone()
            self.actions[env_ids, :] = torch.zeros((len(env_ids), self.num_actions), device = self.device, dtype = torch.float)

            self.progress_buf[env_ids] = 0
            
            self.calculateStateObservations()
            if enable_vis_obs:
                if self.no_vis_grad:
                    with torch.no_grad():
                        pixels = self.render(env_ids)
                        self.vis_obs_buf[env_ids] = torch.tile(torch.moveaxis(pixels, 3, 1), (1, 3, 1, 1))
                else:
                    pixels = self.render(env_ids)
                    self.vis_obs_buf[env_ids] = torch.tile(torch.moveaxis(pixels, 3, 1), (1, 3, 1, 1))
            
        obs = {"state_obs": self.state_obs_buf}
        if enable_vis_obs:
            obs["vis_obs"] = self.vis_obs_buf

        return obs
    
    '''
    This function returns joint_q in mujoco conventions
    '''
    def get_mujoco_joint_q(self, dflex_q:torch.Tensor):
        mujoco_q = dflex_q.clone()
        return mujoco_q

    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self, checkpoint = None):
        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint['joint_q'] = self.state.joint_q.clone()
                checkpoint['joint_qd'] = self.state.joint_qd.clone()
                checkpoint['actions'] = self.actions.clone()
                checkpoint['progress_buf'] = self.progress_buf.clone()

            current_joint_q = checkpoint['joint_q'].clone()
            current_joint_qd = checkpoint['joint_qd'].clone()
            self.state = self.model.state()
            self.state.joint_q = current_joint_q
            self.state.joint_qd = current_joint_qd
            self.actions = checkpoint['actions'].clone()
            self.progress_buf = checkpoint['progress_buf'].clone()
            self.vis_obs_buf = self.vis_obs_buf.clone()

    '''
    This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and it returns the observation vectors
    '''
    def initialize_trajectory(self, enable_vis_obs=False):
        self.clear_grad()
        self.calculateStateObservations()
        obs = {"state_obs": self.state_obs_buf}
        # visual obs already don't have gradient
        if enable_vis_obs:
            obs["vis_obs"] = self.vis_obs_buf

        return obs

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint['joint_q'] = self.state.joint_q.clone()
        checkpoint['joint_qd'] = self.state.joint_qd.clone()
        checkpoint['actions'] = self.actions.clone()
        checkpoint['progress_buf'] = self.progress_buf.clone()

        return checkpoint

    """
    This function calculate the low-dimension state related observation such as velocity in world frame

    The resulted state_obs_buf is fully differentialable c
    """
    def calculateStateObservations(self):
        self.state_obs_buf = torch.cat([self.state.joint_q.view(self.num_envs, -1)[:, 1:], self.state.joint_qd.view(self.num_envs, -1)], dim = -1)

    """
    This function calculate visual observations for given env_ids
    
    Each visual observation is a stack of three images from [t_2, t_1] to current 

    """
    def calculateVisualObservations(self, env_ids):
        pixels = torch.moveaxis(self.render(env_ids), 3, 1)
        old_frames = self.vis_obs_buf[env_ids, 3:, :, :]
        new_vis_buf = torch.cat([old_frames, pixels], dim=1)
        self.vis_obs_buf[env_ids] = new_vis_buf

    def calculateReward(self):
        height_diff = self.state_obs_buf[:, 0] - (self.termination_height + self.termination_height_tolerance)
        height_reward = torch.clip(height_diff, -1.0, 0.3)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)
        
        angle_reward = 1. * (-self.state_obs_buf[:, 1] ** 2 / (self.termination_angle ** 2) + 1.)

        progress_reward = self.state_obs_buf[:, 5]

        self.rew_buf = progress_reward + height_reward + angle_reward + torch.sum(self.actions ** 2, dim = -1) * self.action_penalty
        
        # reset agents
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        if self.early_termination:
            self.reset_buf = torch.where(self.state_obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)