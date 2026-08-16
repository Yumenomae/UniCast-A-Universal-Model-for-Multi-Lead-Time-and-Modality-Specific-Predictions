import math
import torch
import numpy as np
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_, Mlp
from xformers.ops import memory_efficient_attention, unbind
# from .weather_embedding import WeatherEmbedding
from .layers import DownSample3D, FuserLayer, UpSample3D
from .utils import (
    PatchEmbed3D,
    PatchEmbed2D,
    PatchRecovery3D,
    PatchRecovery2D,
)

t_binary_code_map = {1: [1,0,0,0], 3: [1,1,0,0], 6: [0,1,1,0], 12: [0,0,1,1]}
t_binary_code_map = {keys: torch.Tensor(values) for keys, values in t_binary_code_map.items()}

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=4, hidden_features=hidden_size, out_features=hidden_size, act_layer=approx_gelu, drop=0)

    def forward(self, x, t):   # (B)
        t_binary_code = torch.stack([t_binary_code_map[int(lead_time)] for lead_time in t], dim=0).to(device=x.device, dtype=x.dtype) # (B, 5)
        return self.mlp(t_binary_code)


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    An transformers block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class UniCast(nn.Module):
    def __init__(self, 
        in_img_size,
        variables,
        EmbedSet,
        patch_size=5,
        hidden_size=1024,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        depth=(2, 4, 4, 2),
    ):
        super().__init__()

        self.patch_size = patch_size
        
        patch_size = (2, patch_size, patch_size)

        if in_img_size[0] % patch_size[1] != 0:
            pad_size = patch_size[1] - in_img_size[0] % patch_size[1]
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])

        self.in_img_size = in_img_size # (445, 845)

        self.variables = variables

        self.EmbedSet = EmbedSet

        # landtype = torch.from_numpy(np.load(landtype_root)).unsqueeze(0).int() # (1, 441, 845)
        # self.landtype = torch.nn.functional.pad(landtype, (0, 0, pad_size//2, pad_size//2), 'replicate')[0] # (445, 845)
        
        # self.land_embed = nn.Parameter(torch.eye(21).unsqueeze(0), requires_grad=True)

        self.patchembed2d = PatchEmbed2D(
            img_size=in_img_size,
            patch_size=patch_size[1:],
            in_chans=8 + 9,  # add
            embed_dim=hidden_size,
        )
        self.patchembed3d = PatchEmbed3D(
            img_size=(13, in_img_size[0], in_img_size[1]),
            patch_size=patch_size,
            in_chans=5,
            embed_dim=hidden_size,
        )
        patched_inp_shape = (
            8,
            math.ceil(in_img_size[0] / patch_size[1]),
            math.ceil(in_img_size[1] / patch_size[2]),
        )

        self.layer1 = FuserLayer(
            dim=hidden_size,
            input_resolution=patched_inp_shape,
            depth=depth[0],
            num_heads=num_heads[0],
            window_size=window_size,
        )
        patched_inp_shape_downsample = (
            8,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )
        self.downsample = DownSample3D(
            in_dim=hidden_size,
            input_resolution=patched_inp_shape,
            output_resolution=patched_inp_shape_downsample,
        )
        self.layer2 = FuserLayer(
            dim=hidden_size * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=depth[1],
            num_heads=num_heads[1],
            window_size=window_size,
        )
        self.layer3 = FuserLayer(
            dim=hidden_size * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=depth[2],
            num_heads=num_heads[2],
            window_size=window_size,
        )
        self.upsample = UpSample3D(
            hidden_size * 2, hidden_size, patched_inp_shape_downsample, patched_inp_shape
        )
        self.layer4 = FuserLayer(
            dim=hidden_size,
            input_resolution=patched_inp_shape,
            depth=depth[3],
            num_heads=num_heads[3],
            window_size=window_size,
        )

        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = PatchRecovery2D(
            in_img_size, patch_size[1:], 2 * hidden_size, 8
        )
        self.patchrecovery3d = PatchRecovery3D(
            (13, in_img_size[0], in_img_size[1]), patch_size, 2 * hidden_size, 5
        )
        
        # lead time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

    def forward(self, x, hour_of_year, lead_times, variables):
        
        # land_embedding = self.land_embed[:, :, self.landtype-1].repeat(x.shape[0], 1, 1, 1).to(device=x.device, dtype=x.dtype) # (B, 21, H, W)
        time_pos_embedding = torch.stack(
            [self.EmbedSet[int(hour_of_year[i])][0] for i in range(len(hour_of_year))], dim=0).to(device=x.device, dtype=x.dtype
                                                                                               ) # (B, 9, H, W)
        time_pos_embedding = torch.nn.functional.pad(time_pos_embedding, (0, 0, 2, 2), 'replicate')

        surface = x[:, :8, :, :]
        # surface = torch.concat([surface, land_embedding, time_pos_embedding], dim=1)
        surface = torch.concat([surface, time_pos_embedding], dim=1)

        upper_air = x[:, 8:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        lead_times_emb = self.t_embedder(x, lead_times)

        x = self.layer1(x, lead_times_emb)

        skip = x

        x, lead_times_emb = self.downsample(x, lead_times_emb)
        x = self.layer2(x, lead_times_emb)
        x = self.layer3(x, lead_times_emb)
        x, lead_times_emb = self.upsample(x, lead_times_emb)
        x = self.layer4(x, lead_times_emb)
        
        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)

        pred = torch.concat([output_surface, output_upper_air.flatten(1,2)], dim=1)
        return pred

    def first_stage_forward(self, x, hour_of_year, lead_times, variables):

        # land_embedding = self.land_embed[:, :, self.landtype-1].repeat(x.shape[0], 1, 1, 1).to(device=x.device, dtype=x.dtype) # (B, 21, H, W)
        time_pos_embedding = torch.stack(
            [self.EmbedSet[int(hour_of_year[i])][0] for i in range(len(hour_of_year))], dim=0).to(device=x.device, dtype=x.dtype
                                                                                               ) # (B, 9, H, W)
        time_pos_embedding = torch.nn.functional.pad(time_pos_embedding, (0, 0, 2, 2), 'replicate')

        surface = x[:, :8, :, :]
        # surface = torch.concat([surface, land_embedding, time_pos_embedding], dim=1)
        surface = torch.concat([surface, time_pos_embedding], dim=1)

        upper_air = x[:, 8:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        lead_times_emb = self.t_embedder(x, lead_times)

        return x, lead_times_emb, (B, C, Pl, Lat, Lon)