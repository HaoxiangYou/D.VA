import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
import torch
from robot_renderer.base_renderer import BaseRoboRenderer
from utils.background_utils import create_sky_plane, create_floor
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, look_at_view_transform
)

class CheetahRenderer(BaseRoboRenderer):
    def __init__(self, img_height=84, img_width=84, device="cuda"):
        super().__init__(os.path.join(proj_dir, "assets/xmls/half_cheetah.xml"), img_height=img_height, img_width=img_width, device=device)

    def build_background(self):
        self.raw_meshes["background"] = [create_floor(x_dim=150, y_dim=2, device=self.device, center=[10, 0, -0.35]),
                                        create_sky_plane(center=[10, 1, 3.25], z_dim=10, x_dim=150, device=self.device)
                                        ]
        
    def get_camera(self, qpos, camera_id):
        qpos = torch.atleast_2d(qpos)
        batch_size = qpos.shape[0]
        pos = torch.concat([qpos[:, 0:1], torch.zeros_like(qpos[:, 0:1]), qpos[:, 1:2]], dim=1)
        R, T = look_at_view_transform(
            eye=torch.tile(torch.tensor([0, -3, 0.25], dtype=torch.float32, device=self.device), (batch_size, 1)) + pos,
            at=torch.tile(torch.tensor([0, 0, 0.25], dtype=torch.float32, device=self.device), (batch_size, 1)) + pos,
            up=torch.tile(torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device), (batch_size, 1)),
            device=self.device)
                
        return FoVPerspectiveCameras(znear=0.01, zfar=100, fov=45, R=R, T=T, device=self.device)

    def get_raster_settings(self, camera_id):
        if camera_id == 0:
            # img size for training
            return RasterizationSettings(image_size=(self.img_height, self.img_width))
        else:
            # img size for rendering
            return RasterizationSettings(image_size=256)
        
    def get_lights(self, qpos):
        qpos = torch.atleast_2d(qpos)
        batch_size = qpos.shape[0]
        
        light_position = torch.concat([qpos[:, 0:1], torch.zeros_like(qpos[:, 0:1]), qpos[:, 1:2]], dim=1) + torch.tensor([[0.0, 0.0, 2.0]], device=self.device).expand(batch_size, -1)

        # Define the light parameters
        ambient_color = torch.tensor([[0.4, 0.4, 0.4]], device=self.device).expand(batch_size, -1)   # Ambient light color
        diffuse_color = torch.tensor([[0.8, 0.8, 0.8]], device=self.device).expand(batch_size, -1)   # Diffuse light color
        specular_color = torch.tensor([[0.1, 0.1, 0.1]], device=self.device) .expand(batch_size, -1) # Specular highlights

        # Create the PointLight object in PyTorch3D
        return PointLights(
            device=self.device,
            location=light_position,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color
        )
        
    def get_transform(self, qpos):
        qpos = torch.atleast_2d(qpos)
        return self.chain.forward_kinematics(qpos)

    def get_renderer(self, qpos, camera_id):
        cameras = self.get_camera(qpos, camera_id)
        raster_settings = self.get_raster_settings(camera_id)
        lights = self.get_lights(qpos)

        return MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings),
                shader=SoftPhongShader(cameras=cameras, device=self.device, lights=lights)
            )