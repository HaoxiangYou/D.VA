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

class HopperVis(BaseAgent):
    uid = "hopper_vis"
    mjcf_path = os.path.join(os.path.dirname(__file__), "assets/hopper.xml")
    
@register_env("HopperVis")
class HopperVisEnv(BaseEnv):
    agent: Union[HopperVis]

    def __init__(self, *args, img_width=84, img_height=84, robot_uids=HopperVis, **kwargs):
        self.img_width=img_width
        self.img_height=img_height
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                uid="recording_cam",
                pose=sapien_utils.look_at(eye=[0, -2.8, 0.8], target=[0, 0, 0]),
                width=256,
                height=256,
                fov=np.pi / 4,
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["torso_dummy_1"],
            ),
        ]
    
    @property
    def _default_human_render_camera_configs(self):
        return [
            CameraConfig(
                uid="training_cam",
                pose=sapien_utils.look_at(eye=[0, -2.8, 0.8], target=[0, 0, 0]),
                width=self.img_width,
                height=self.img_height,
                fov=np.pi / 4,
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["torso_dummy_1"],
            ),
        ]

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        actor_builders = loader.parse(os.path.join(os.path.dirname(__file__), "assets/hopper.xml"))["actor_builders"]
        for a in actor_builders:
            a.build(a.name)

        self.planar_scene = BackgroundSceneBuilder(env=self)
        self.planar_scene.build(floor_length=1000, floor_width=2, altitude=0)

    def update_vis(self, qpos):
        self.agent.robot.set_qpos(qpos)
        self.scene.px.gpu_apply_articulation_qpos()
        self.scene.px.gpu_update_articulation_kinematics()
        self.scene._gpu_fetch_all()