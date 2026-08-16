# Standard library
import os

# Third-party
import numpy as np
import torch


def main():
    """
    Pre-compute all static features related to the grid nodes
    """
    lat = np.load('/home-ssd/Users/gm_lhy/zhengjq/Ours/data/lat.npy')
    lon = np.load('/home-ssd/Users/gm_lhy/zhengjq/Ours/data/lon.npy')

    xx, yy = np.meshgrid(lon, lat)
    xy = np.stack((xx, yy))

    # -- Static grid node features --
    grid_xy = torch.tensor(xy)  # (2, N_x, N_y)
    grid_xy = grid_xy.flatten(1, 2).T  # (N_grid, 2)
    pos_max = torch.max(torch.abs(grid_xy))
    grid_xy = grid_xy / pos_max  # Divide by maximum coordinate

    # Concatenate grid features
    grid_features = grid_xy

    torch.save(grid_features, './src/models/components/GraphFM/grid_features.pt')


if __name__ == "__main__":
    main()