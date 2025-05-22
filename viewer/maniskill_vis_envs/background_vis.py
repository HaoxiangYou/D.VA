from typing import List
import sapien
import numpy as np
from transforms3d.euler import euler2quat
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


class BackgroundSceneBuilder(SceneBuilder):
    def build(self, floor_width=2, floor_length=100, altitude=0, xy_origin=(0,0)):
        # ground - a strip with length along +x
        self.ground = build_ground(
            self.scene,
            floor_width=floor_width,
            floor_length=floor_length,
            altitude=altitude,
            xy_origin=xy_origin,
            add_collision=False
        )

        # background visual wall
        self.wall = self.scene.create_actor_builder()
        self.wall.add_box_visual(
            half_size=(1e-3, floor_length, 20),
            pose=sapien.Pose(p=[xy_origin[0], floor_width, altitude], q=euler2quat(0, 0, np.pi / 2)),
            material=sapien.render.RenderMaterial(
                base_color=np.array([0.3, 0.3, 0.3, 1])
            ),
        )
        self.wall.build_static(name="wall")
        self.scene_objects: List[sapien.Entity] = [self.ground, self.wall]