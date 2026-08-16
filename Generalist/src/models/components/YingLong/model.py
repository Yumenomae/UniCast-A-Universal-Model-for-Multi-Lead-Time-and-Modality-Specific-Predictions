# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import torch
import torch.nn as nn

from .layers import DownSample3D, FuserLayer, UpSample3D
from .utils import (
    PatchEmbed3D,
    PatchEmbed2D,
    PatchRecovery3D,
    PatchRecovery2D,
)


class YingLong(nn.Module):

    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_blocks=16,
        depth=12,
        window_size=(2, 6, 12),
    ):
        super().__init__()
        # drop_path = np.linspace(0, 0.2, 8).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        self.patchembed2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size[1:],
            in_chans=8, 
            embed_dim=embed_dim,
        )
        self.patchembed3d = PatchEmbed3D(
            img_size=(13, img_size[0], img_size[1]),
            patch_size=patch_size,
            in_chans=5,
            embed_dim=embed_dim,
        )
        patched_inp_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        self.layer1 = FuserLayer(
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=depth,
            num_heads=num_blocks,
            num_blocks=num_blocks,
            window_size=window_size,
        )

        self.patchrecovery2d = PatchRecovery2D(
            img_size, patch_size[1:], embed_dim, 8
        )
        self.patchrecovery3d = PatchRecovery3D(
            (13, img_size[0], img_size[1]), patch_size, embed_dim, 5
        )


    def forward(self, x, hour_of_year):
        """
        Args:
            x (torch.Tensor): [batch, 4+3+5*13, lat, lon]
        """

        surface = x[:, :8, :, :]
        surface = torch.concat([surface], dim=1)

        upper_air = x[:, 8:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        x = self.layer1(x)

        output = x
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)

        pred = torch.concat([output_surface, output_upper_air.flatten(1,2)], dim=1)
        return pred