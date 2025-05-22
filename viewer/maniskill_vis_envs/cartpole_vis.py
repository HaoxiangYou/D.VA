import os, sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_dir)

import numpy as np
from typing import Union
from viewer.maniskill_vis_envs.background_vis import BackgroundSceneBuilder
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils

class CartpoleVis(BaseAgent):
    uid = "cartpole_vis"
    mjcf_path = os.path.join(os.path.dirname(__file__), "assets/cartpole.xml")

@register_env("CartpoleVis")
class CartpoleVisEnv(BaseEnv):
    agent: Union[CartpoleVis]

    def __init__(self, *args, img_width=84, img_height=84, robot_uids=CartpoleVis, **kwargs):
        self.img_width=img_width
        self.img_height=img_height
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                uid="recording_cam", 
                pose=sapien_utils.look_at(eye=[0, -2.8, 1.2], target=[0, 0, 1.2]), 
                width=256, 
                height=256, 
                fov=np.pi/2, 
                near=0.01, 
                far=100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        return [
            CameraConfig(
                uid="training_cam", 
                pose=sapien_utils.look_at(eye=[0, -2.8, 1.2], target=[0, 0, 1.2]), 
                width=self.img_width,
                height=self.img_height, 
                fov=np.pi/2, 
                near=0.01, 
                far=100),]

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        actor_builders = loader.parse(os.path.join(os.path.dirname(__file__), "assets/cartpole.xml"))["actor_builders"]
        for a in actor_builders:
            a.build(a.name)
        self.planar_scene = BackgroundSceneBuilder(env=self)
        self.planar_scene.build(floor_length=20, floor_width=10, altitude=0)

    def update_vis(self, qpos):
        self.agent.robot.set_qpos(qpos)
        self.scene.px.gpu_apply_articulation_qpos()
        self.scene.px.gpu_update_articulation_kinematics()
        self.scene._gpu_fetch_all()