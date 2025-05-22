import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)

import mujoco
import torch
from kinematics.chain import Chain
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
import trimesh

# Types we can convert to meshes
GEOM_TYPE_MAP = {
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
}

def build_pytorch3d_mesh(geom, texture_res=10, margin=0.3, device='cpu'):
    geom_type = GEOM_TYPE_MAP.get(geom.type_id, None)
    size = geom.size
    rgb = torch.tensor(geom.rgba[:3], dtype=torch.float32, device=device)

    # Create base geometry using trimesh
    if geom_type == "box":
        half_extents = size[:3]
        trmesh = trimesh.creation.box(extents=2 * half_extents)
    elif geom_type == "capsule":
        radius, half_height = size[0], size[1]
        trmesh = trimesh.creation.capsule(radius=radius, height=2 * half_height, count=[16, 8])
    elif geom_type == "sphere":
        trmesh = trimesh.creation.icosphere(radius=size[0])
    elif geom_type == "cylinder":
        radius, half_height = size[0], size[1]
        trmesh = trimesh.creation.cylinder(radius=radius, height=2 * half_height)
    else:
        raise ValueError("Unsupported geom!")

    # === Convert to torch tensors ===
    verts = torch.tensor(trmesh.vertices, dtype=torch.float32, device=device)
    verts = geom.offset.transform_points(verts)[0]  # Apply offset transform
    faces = torch.tensor(trmesh.faces, dtype=torch.int64, device=device)

    # === Dummy UVs: normalize XY to [margin, 1 - margin] ===
    verts_uvs = verts[:, :2]

    min_uv = verts_uvs.min(dim=0)[0]
    max_uv = verts_uvs.max(dim=0)[0]
    scale = max_uv - min_uv + 1e-6

    verts_uvs = (verts_uvs - min_uv) / scale  # Normalize to [0, 1]
    verts_uvs = verts_uvs * (1 - 2 * margin) + margin  # Scale to [margin, 1 - margin]
    verts_uvs = verts_uvs.clamp(0, 1).unsqueeze(0)  # (1, V, 2)

    faces_uvs = faces.clone().unsqueeze(0)  # (1, F, 3)

    # === Texture image: solid RGB tile ===
    tex_img = rgb.view(1, 1, 1, 3).expand(1, texture_res, texture_res, 3)  # (1, H, W, 3)

    # === Assemble TexturesUV ===
    textures = TexturesUV(
        maps=tex_img,
        faces_uvs=faces_uvs,
        verts_uvs=verts_uvs
    )

    # === Final Mesh ===
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return mesh

def build_meshes_from_chain(chain: Chain):
    frames_name = chain.get_frame_names(exclude_fixed_and_free=False)
    body_to_mesh = {}
    for frame_name in frames_name:
        geoms = []
        frame = chain.find_frame(frame_name)
        for geom in frame.geoms:
            geoms.append(build_pytorch3d_mesh(geom))
        body_to_mesh[frame_name] = geoms
    
    return body_to_mesh    