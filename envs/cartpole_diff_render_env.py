from envs.dflex_diff_render_env import DFlexDiffRenderEnv
import math
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

from utils import load_utils as lu
from utils import torch_utils as tu
from viewer.torch3d_robo import CartpoleRenderer

class CartPoleDiffRenderEnv(DFlexDiffRenderEnv):

    def __init__(self, device='cuda:0', num_envs=1024, seed=0, episode_length=240, img_height=84, img_width=84, no_grad=True, 
                no_vis_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = True):

        num_state_obs = 5
        num_act = 1

        super(CartPoleDiffRenderEnv, self).__init__(num_envs, num_state_obs, num_act, episode_length, MM_caching_frequency, seed, 
                                    no_grad=no_grad, no_vis_grad=no_vis_grad, device=device, img_height=img_height, img_width=img_width)

        self.stochastic_init = stochastic_init
        self.early_termination = early_termination

        self.init_sim()

        self.renderer = CartpoleRenderer(img_height=img_height, img_width=img_width, device=device)

        # boundary 
        self.termination_pos = 2.5

        # action parameters
        self.action_strength = 1000.

        # loss related
        self.health_offset = 10.
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1

        self.cart_position_penalty = 0.05
        self.cart_velocity_penalty = 0.1

        self.cart_action_penalty = 0.0

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1. / 60.
        self.sim_substeps = 4
        self.sim_dt = self.dt

        self.num_joint_q = 2
        self.num_joint_qd = 2

        asset_folder = os.path.join(os.path.dirname(__file__), 'assets')        
        for i in range(self.num_environments):
            lu.urdf_load(self.builder, 
                                os.path.join(asset_folder, 'cartpole.urdf'),
                                df.transform((0.0, 2.5, 0.0), df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)), 
                                floating=False,
                                shape_kd=1e4,
                                limit_kd=1.)
            self.builder.joint_q[i * self.num_joint_q + 1] = -math.pi
        
        self.model = self.builder.finalize(self.device)
        self.model.ground = False
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype = torch.float, device = self.device)

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()
        self.start_joint_q = self.state.joint_q.clone()
        self.start_joint_qd = self.state.joint_qd.clone()

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
        
        self.state.joint_act.view(self.num_envs, -1)[:, 0:1] = actions * self.action_strength
        
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
                self.reset(env_ids=env_ids, enable_vis_obs=enable_vis_obs)

        obs = {"state_obs": self.state_obs_buf}
        if enable_vis_obs:
            obs["vis_obs"] = self.vis_obs_buf
            
        return obs, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self, env_ids=None, force_reset=True, enable_vis_obs=False):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # fixed start state
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.view(-1, self.num_joint_q)[env_ids, :].clone()
            self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = self.start_joint_qd.view(-1, self.num_joint_qd)[env_ids, :].clone()

            if self.stochastic_init:
                self.state.joint_q.view(self.num_envs, -1)[env_ids, :] = \
                    self.state.joint_q.view(self.num_envs, -1)[env_ids, :] \
                    + np.pi / 6 * (torch.rand(size=(len(env_ids), self.num_joint_q), device=self.device) - 0.5)

                self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = \
                    self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] \
                    + 0.1 * (torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5)
            
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
    def clear_grad(self):
        with torch.no_grad(): # TODO: check with Miles
            current_joint_q = self.state.joint_q.clone()
            current_joint_qd = self.state.joint_qd.clone() 
            current_joint_act = self.state.joint_act.clone()
            self.state = self.model.state()
            self.state.joint_q = current_joint_q
            self.state.joint_qd = current_joint_qd
            self.state.joint_act = current_joint_act
            self.vis_obs_buf = self.vis_obs_buf.clone()

    '''
    This function starts collecting a new trajectory from the current states but cut off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and return the observation vectors
    '''
    def initialize_trajectory(self, enable_vis_obs=False):
        self.clear_grad()
        self.calculateStateObservations()
        obs = {"state_obs": self.state_obs_buf}
        if enable_vis_obs:
            obs["vis_obs"] = self.vis_obs_buf

        return obs

    def calculateStateObservations(self):
        x = self.state.joint_q.view(self.num_envs, -1)[:, 0:1]
        theta = self.state.joint_q.view(self.num_envs, -1)[:, 1:2]
        xdot = self.state.joint_qd.view(self.num_envs, -1)[:, 0:1]
        theta_dot = self.state.joint_qd.view(self.num_envs, -1)[:, 1:2]

        # observations: [x, xdot, sin(theta), cos(theta), theta_dot]
        self.state_obs_buf = torch.cat([x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim = -1)

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
        x = self.state.joint_q.view(self.num_envs, -1)[:, 0]
        theta = tu.normalize_angle(self.state.joint_q.view(self.num_envs, -1)[:, 1])
        xdot = self.state.joint_qd.view(self.num_envs, -1)[:, 0]
        theta_dot = self.state.joint_qd.view(self.num_envs, -1)[:, 1]

        self.rew_buf = self.health_offset \
                    -torch.pow(theta, 2.) * self.pole_angle_penalty \
                    - torch.pow(theta_dot, 2.) * self.pole_velocity_penalty \
                    - torch.pow(x, 2.) * self.cart_position_penalty \
                    - torch.pow(xdot, 2.) * self.cart_velocity_penalty \
                    - torch.sum(self.actions ** 2, dim = -1) * self.cart_action_penalty
        
        # reset agents
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        if self.early_termination:
            self.reset_buf = torch.where(
                (self.state_obs_buf[:, 0] > self.termination_pos) | (self.state_obs_buf[:, 0] < -self.termination_pos),
                torch.ones_like(self.reset_buf),
                self.reset_buf
            )