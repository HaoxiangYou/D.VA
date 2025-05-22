import os, sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

import viewer.maniskill_vis_envs
import gymnasium as gym
import torch

class ManiskillViewer:
    def __init__(self, env_name, num_env, height=84, width=84):
        self.env_name = env_name
        self.num_env = num_env
        self.height = height
        self.width = width

        self.env = gym.make(
            id=self.env_name,
            num_envs=self.num_env,
            img_height=self.height,
            img_width=self.width,
            obs_mode="rgb",
            reward_mode="none",
            render_mode="none",
            enable_shadow=True,
            sim_backend="gpu",
            render_backend="gpu",
            sensor_configs=dict(shader_pack="default"),
        )

    def render(self, qpos:torch.Tensor, recording=False):
        """
        qpos: joint qpos in mujoco conventions
        """
        self.env.update_vis(qpos)
        if recording:
            pixels = self.env.render_sensors()
        else:
            pixels = self.env.render_rgb_array()
        
        return pixels
