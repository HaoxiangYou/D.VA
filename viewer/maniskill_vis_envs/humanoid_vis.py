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
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import sapien_utils

class HumanoidVis(BaseAgent):
    uid = "humanoid_vis"
    mjcf_path = os.path.join(os.path.dirname(__file__), "assets/humanoid.xml")
    fix_root_link = False

@register_env("HumanoidVis")
class HumanoidVisEnv(BaseEnv):
    agent: Union[HumanoidVis]

    maniskill_joint_index = ['abdomen_z', 'right_shoulder1', 'left_shoulder1', 'abdomen_y', 'right_shoulder2', 'left_shoulder2', 'abdomen_x', 
                            'right_elbow', 'left_elbow', 'right_hip_x', 'left_hip_x', 'right_hip_z', 'left_hip_z', 'right_hip_y', 
                            'left_hip_y', 'right_knee', 'left_knee', 'right_ankle_y', 'left_ankle_y', 'right_ankle_x', 'left_ankle_x']
    
    mujoco_joint_index = ["abdomen_z", "abdomen_y", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee", 
                          "right_ankle_y", "right_ankle_x", "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "left_ankle_y",
                          "left_ankle_x", "right_shoulder1", "right_shoulder2", "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow"]
    
    def __init__(self, *args, img_width=84, img_height=84, robot_uids=HumanoidVis, **kwargs):
        self.img_width=img_width
        self.img_height=img_height
        self.mujoco_joint_to_maniskill_map = [self.mujoco_joint_index.index(item) for item in self.maniskill_joint_index]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                uid="recording_cam",
                pose=sapien_utils.look_at(eye=[-3, 0, 1], target=[0, 0, 0]),
                width=256,
                height=256,
                fov=np.pi / 4,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            ),
        ]
    
    @property
    def _default_human_render_camera_configs(self):
        return [
            CameraConfig(
                uid="training_cam",
                pose=sapien_utils.look_at(eye=[-3, 0, 1], target=[0, 0, 0]),
                width=self.img_width,
                height=self.img_height,
                fov=np.pi / 4,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            ),
        ]

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        actor_builders = loader.parse(os.path.join(os.path.dirname(__file__), "assets/humanoid.xml"))["actor_builders"]
        for a in actor_builders:
            a.build(a.name)

        self.planar_scene = BackgroundSceneBuilder(env=self)
        self.planar_scene.build(floor_width=50, floor_length=1000, altitude=0)

        # allow tracking of humanoid
        self.camera_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")
    
    def update_vis(self, qpos):
        assert len(qpos.shape) == 2
        pos = qpos[:, :3]
        quat = qpos[:, 3:7]

        self.agent.robot.set_root_pose(Pose.create_from_pq(p=pos, q=quat))
        self.agent.robot.set_qpos(qpos[:, 7:][:, self.mujoco_joint_to_maniskill_map])
        self.camera_mount.set_pose(
            Pose.create_from_pq(p=pos)
        )
        self.scene.px.gpu_apply_rigid_dynamic_data()
        self.scene.px.gpu_apply_articulation_root_pose()
        self.scene.px.gpu_apply_articulation_qpos()
        self.scene.px.gpu_update_articulation_kinematics()
        self.scene._gpu_fetch_all()