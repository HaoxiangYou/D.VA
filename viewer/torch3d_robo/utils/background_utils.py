import os, sys
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
import torch
from pytorch3d.structures import Meshes
import imageio.v3 as iio
from pytorch3d.renderer import TexturesUV

def create_floor(center=[0, 0, 0], x_dim=200.0, y_dim=200.0, margin=0, res=2,
                          texture_image_path=os.path.join(proj_dir, 'assets/textures/checkerboard.png'), device='cpu'):
    cx, cy, cz = center
    hx, hy = x_dim / 2, y_dim / 2

    # Define 4 vertices (XY plane at Z=cz)
    verts = torch.tensor([
        [cx - hx, cy - hy, cz],
        [cx + hx, cy - hy, cz],
        [cx + hx, cy + hy, cz],
        [cx - hx, cy + hy, cz],
    ], dtype=torch.float32, device=device)

    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.int64, device=device)

    # UVs repeat the texture tile_x x tile_y times
    uvs = torch.tensor([
        [margin, margin],
        [margin, 1-margin],
        [1-margin, 1-margin],
        [1-margin, margin],
    ], dtype=torch.float32, device=device)

    face_uvs = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.int64, device=device)

    # Load the texture image
    image = iio.imread(texture_image_path)
    if image.max() > 1:
        image = image / 255.0
    image_h, image_w = image.shape[:2]

    image = torch.tensor(image, dtype=torch.float32, device=device)[int((image_h - x_dim * res)/2):int((image_h + x_dim * res)/2), 
                                                                    int((image_w - y_dim * res)/2):int((image_w + y_dim * res)/2)].unsqueeze(0)  # (1, H, W, 3)

    textures = TexturesUV(maps=image, faces_uvs=[face_uvs], verts_uvs=[uvs])
    return Meshes(verts=[verts], faces=[faces], textures=textures)

def create_sky_plane(center=[0, 0, 0], x_dim=200.0, z_dim=100.0, margin=0, res=2,
                     texture_image_path=os.path.join(proj_dir, 'assets/textures/skybox.png'),
                     device='cpu'):
    cx, cy, cz = center
    hx, hz = x_dim / 2, z_dim / 2

    # Define 4 vertices (XZ plane at Y=cy)
    verts = torch.tensor([
        [cx - hx, cy, cz - hz],
        [cx + hx, cy, cz - hz],
        [cx + hx, cy, cz + hz],
        [cx - hx, cy, cz + hz],
    ], dtype=torch.float32, device=device)

    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.int64, device=device)

    uvs = torch.tensor([
        [margin, margin],
        [margin, 1-margin],
        [1-margin, 1-margin],
        [1-margin, margin],
    ], dtype=torch.float32, device=device)

    face_uvs = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.int64, device=device)

    # Load the texture image
    image = iio.imread(texture_image_path)
    if image.max() > 1:
        image = image / 255.0
    image_h, image_w = image.shape[:2]

    image = torch.tensor(image, dtype=torch.float32, device=device)[int((image_h - x_dim * res)/2):int((image_h + x_dim * res)/2), 
                                                                    int((image_w - z_dim * res)/2):int((image_w + z_dim * res)/2)].unsqueeze(0)  # (1, H, W, 3)

    textures = TexturesUV(maps=image, faces_uvs=[face_uvs], verts_uvs=[uvs])
    return Meshes(verts=[verts], faces=[faces], textures=textures)
