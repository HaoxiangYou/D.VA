import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV

def fuse_textures(texture_list):
    """
    Fast, batched version to fuse a list of TexturesUV (with same batch size).
    Fuses maps side-by-side horizontally and adjusts UVs per batch in parallel.
    
    Returns a single TexturesUV object with fused maps, verts_uvs, faces_uvs.
    """

    B = texture_list[0].maps_padded().shape[0]
    assert all(t.maps_padded().shape[0] == B for t in texture_list), "Batch sizes must match"

    maps_list = [t.maps_padded() for t in texture_list]            # List of (B, H_i, W_i, 3)
    verts_uvs_list = [t.verts_uvs_padded() for t in texture_list]  # List of (B, Vt_i, 2)
    faces_uvs_list = [t.faces_uvs_padded() for t in texture_list]  # List of (B, F_i, 3)

    heights = [m.shape[1] for m in maps_list]
    widths = [m.shape[2] for m in maps_list]
    max_H = max(heights)
    total_W = sum(widths)

    fused_maps = []
    fused_verts_uvs = []
    fused_faces_uvs = []

    vt_offset = 0
    width_offset = 0

    for i in range(len(texture_list)):
        m = maps_list[i]      # (B, H, W, 3)
        vt = verts_uvs_list[i]  # (B, Vt, 2)
        fvt = faces_uvs_list[i]  # (B, F, 3)

        H, W = m.shape[1:3]

        # Pad to max height
        pad_top = 0
        pad_bottom = max_H - H
        m_padded = F.pad(m, (0, 0, 0, 0, pad_top, pad_bottom))  # (B, max_H, W, 3)
        fused_maps.append(m_padded)

        # Adjust verts_uvs
        uvs = vt.clone()
        uvs[..., 0] = (width_offset + W * uvs[..., 0]) / total_W
        uvs[..., 1] = (uvs[..., 1] * H + pad_bottom) / max_H

        fused_verts_uvs.append(uvs)

        # Offset faces_uvs
        fvt_offset = fvt + vt_offset
        fused_faces_uvs.append(fvt_offset)

        # Update offsets
        vt_offset += vt.shape[1]
        width_offset += W

    # Concatenate final tensors
    maps_final = torch.cat(fused_maps, dim=2)                  # (B, H, W_total, 3)
    verts_uvs_final = torch.cat(fused_verts_uvs, dim=1)        # (B, V_total, 2)
    faces_uvs_final = torch.cat(fused_faces_uvs, dim=1)        # (B, F_total, 3)

    return TexturesUV(
        maps=maps_final,
        verts_uvs=verts_uvs_final,
        faces_uvs=faces_uvs_final,
    )

def fuse_meshes(meshes_list: list[Meshes]) -> Meshes:
    """
    Fusing M batched Meshes with TextureUV (N each)
    into N Meshes, each with M parts fused.
    Args:
        meshes_list: list of M Meshes with batch size N
    Returns:
        Meshes with batch size N, each containing M sub-meshes fused
    """
    batch_size = len(meshes_list[0])
    num_objs = len(meshes_list)

    # Step 1: Collect all verts/faces/textures in M x N order
    verts_all = []
    faces_all = []
    textures_all = []

    for mesh_idx, mesh in enumerate(meshes_list):
        verts_list = mesh.verts_list()  # list of N tensors
        faces_list = mesh.faces_list()
        textures_all.append(mesh.textures)

        for i in range(batch_size):
            verts_all.append(verts_list[i])
            faces_all.append(faces_list[i])

    # Step 2: Build big tensors and track offsets
    scene_verts = []
    scene_faces = []

    for i in range(batch_size):
        v_offset = 0
        verts_scene = []
        faces_scene = []

        for j in range(num_objs):
            idx = i + j * batch_size
            verts = verts_all[idx]
            faces = faces_all[idx] + v_offset

            verts_scene.append(verts)
            faces_scene.append(faces)
            v_offset += verts.shape[0]

        scene_verts.append(torch.cat(verts_scene, dim=0))
        scene_faces.append(torch.cat(faces_scene, dim=0))

    # Fuse textures
    textures = fuse_textures(textures_all)
    return Meshes(verts=scene_verts, faces=scene_faces, textures=textures)

