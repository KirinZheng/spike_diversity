# Copyright 2024 The HuggingFace Team. All rights reserved.
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py

from typing import List, Optional, Tuple, Union, Any, Dict
import torch



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, output_type="pt"):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`torch.Tensor`): 1D tensor of positions with shape `(M,)`

    Returns:
        `torch.Tensor`: Sinusoidal positional embeddings of shape `(M, D)`.
    """

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, output_type="pt"):
    r"""
    This function generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid (`torch.Tensor`): Grid of positions with shape `(H * W,)`.

    Returns:
        `torch.Tensor`: The 2D sinusoidal positional embeddings with shape `(H * W, embed_dim)`
    """
    
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], output_type=output_type)  # (H*W, D/2), row
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], output_type=output_type)  # (H*W, D/2), column

    emb = torch.concat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb




def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
    device: Optional[torch.device] = None,
    output_type: str = "pt",
) -> torch.Tensor:
    r"""
    Creates 3D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`):
            The embedding dimension of inputs. It must be divisible by 16.
        spatial_size (`int` or `Tuple[int, int]`):
            The spatial dimension of positional embeddings. If an integer is provided, the same size is applied to both
            spatial dimensions (height and width). spatial_size[0] and [1] refers to width and height.
        temporal_size (`int`):
            The temporal dimension of postional embeddings (number of frames).
        spatial_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for spatial grid interpolation.
        temporal_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for temporal grid interpolation.

    Returns:
        `torch.Tensor`:
            The 3D sinusoidal positional embeddings of shape `[temporal_size, 
            spatial_size[0] * spatial_size[1], embed_dim]`.
    """

    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    # 这里说是spatial的embed dimension取 3/4 大小，时间embed dimension取 1/4，
    # 说是空间信息（图像内容）通常比时间信息（帧间运动）更复杂（类似TimeSformer设计）
    embed_dim_spatial = 3 * embed_dim // 4   
    embed_dim_temporal = embed_dim // 4

    # 1. Spatial
    grid_h = torch.arange(spatial_size[1], device=device, dtype=torch.float32) / spatial_interpolation_scale
    grid_w = torch.arange(spatial_size[0], device=device, dtype=torch.float32) / spatial_interpolation_scale
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first, get grid shape [h, w]
    grid = torch.stack(grid, dim=0) # 2 h w, row + column index

    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])   # 2 1 h w
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid, output_type="pt")        # HW D//4*3

    # 2. Temporal
    grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32) / temporal_interpolation_scale
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t, output_type="pt")    # T D//4

    # 3. Concat
    pos_embed_spatial = pos_embed_spatial[None, :, :]   # 1 HW D
    pos_embed_spatial = pos_embed_spatial.repeat_interleave(
        temporal_size, dim=0, output_size=pos_embed_spatial.shape[0] * temporal_size
    )  # [T, H*W, D // 4 * 3]

    pos_embed_temporal = pos_embed_temporal[:, None, :]
    pos_embed_temporal = pos_embed_temporal.repeat_interleave(
        spatial_size[0] * spatial_size[1], dim=1
    )  # [T, H*W, D // 4]

    pos_embed = torch.concat([pos_embed_temporal, pos_embed_spatial], dim=-1)  # [T, H*W, D]
    return pos_embed




def get_sinusoid_spatial_temporal_encoding(height, width, time_step, d_hid):
    ''' Sinusoid position encoding table '''
    n_position = height * width

    def get_position_angle_vec(position):
        return [position / torch.pow(torch.tensor(10000), 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # Create sinusoid spatial table
    sinusoid_spatial_table = torch.stack([torch.tensor(get_position_angle_vec(pos_i)) for pos_i in range(n_position)])
    sinusoid_spatial_table[:, 0::2] = torch.sin(sinusoid_spatial_table[:, 0::2])  # dim 2i
    sinusoid_spatial_table[:, 1::2] = torch.cos(sinusoid_spatial_table[:, 1::2])  # dim 2i+1

    # Create sinusoid temporal table
    sinusoid_temporal_table = torch.stack([torch.tensor(get_position_angle_vec(pos_i)) for pos_i in range(time_step)])
    sinusoid_temporal_table[:, 0::2] = torch.sin(sinusoid_temporal_table[:, 0::2])  # dim 2i
    sinusoid_temporal_table[:, 1::2] = torch.cos(sinusoid_temporal_table[:, 1::2])  # dim 2i+1

    # Reshape and prepare the spatial and temporal encoding
    sinusoid_spatial = sinusoid_spatial_table.view(height, width, d_hid).unsqueeze(0).contiguous()  # 1 H W C
    sinusoid_temporal = sinusoid_temporal_table[:, None, None, :]  # time_steps 1 1 C

    # Add spatial and temporal encodings
    sinusoid_spatial_temporal = sinusoid_spatial + sinusoid_temporal

    return sinusoid_spatial_temporal.unsqueeze(0).permute(1, 0, 4, 2, 3)  # T 1 C H W, add batch dimension

