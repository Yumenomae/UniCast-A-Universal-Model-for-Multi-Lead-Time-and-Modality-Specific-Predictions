import torch
import torch.nn as nn

def window_partition(x: torch.Tensor, window_size, ndim=3):
    """
    Args:
        x: (B, Pl, Lat, Lon, C) or (B, Lat, Lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon] or [win_lat, win_lon]
        ndim (int): dimension of window (3 or 2)

    Returns:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C) or (B*num_lon, num_lat, win_lat, win_lon, C)
    """
    if ndim == 3:
        B, Pl, Lat, Lon, C = x.shape
        win_pl, win_lat, win_lon = window_size
        x = x.view(
            B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C
        )
        windows = (
            x.permute(0, 5, 1, 3, 2, 4, 6, 7)
            .contiguous()
            .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C)
        )
        return windows
    elif ndim == 2:
        B, Lat, Lon, C = x.shape
        win_lat, win_lon = window_size
        x = x.view(B, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
        windows = (
            x.permute(0, 3, 1, 2, 4, 5)
            .contiguous()
            .view(-1, (Lat // win_lat), win_lat, win_lon, C)
        )
        return windows


def window_reverse(windows, window_size, Pl=1, Lat=1, Lon=1, ndim=3):
    """
    Args:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C) or (B*num_lon, num_lat, win_lat, win_lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon] or [win_lat, win_lon]
        Pl (int): pressure levels
        Lat (int): latitude
        Lon (int): longitude
        ndim (int): dimension of window (3 or 2)

    Returns:
        x: (B, Pl, Lat, Lon, C) or (B, Lat, Lon, C)
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(
            B,
            Lon // win_lon,
            Pl // win_pl,
            Lat // win_lat,
            win_pl,
            win_lat,
            win_lon,
            -1,
        )
        x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
        return x
    elif ndim == 2:
        win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(B, Lon // win_lon, Lat // win_lat, win_lat, win_lon, -1)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, Lat, Lon, -1)
        return x


def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
    """
    Along the longitude dimension, the leftmost and rightmost indices are actually close to each other.
    If half windows apper at both leftmost and rightmost positions, they are dircetly merged into one window.
    Args:
        input_resolution (tuple[int]): [pressure levels, latitude, longitude] or [latitude, longitude]
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude] or [latitude, longitude]
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude] or [latitude, longitude]
        ndim (int): dimension of window (3 or 2)

    Returns:
        attn_mask: (n_lon, n_pl*n_lat, win_pl*win_lat*win_lon, win_pl*win_lat*win_lon) or (n_lon, n_lat, win_lat*win_lon, win_lat*win_lon)
    """
    if ndim == 3:
        Pl, Lat, Lon = input_resolution
        win_pl, win_lat, win_lon = window_size
        shift_pl, shift_lat, shift_lon = shift_size

        img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))
    elif ndim == 2:
        Lat, Lon = input_resolution
        win_lat, win_lon = window_size
        shift_lat, shift_lon = shift_size

        img_mask = torch.zeros((1, Lat, Lon + shift_lon, 1))

    if ndim == 3:
        pl_slices = (
            slice(0, -win_pl),
            slice(-win_pl, -shift_pl),
            slice(-shift_pl, None),
        )
    lat_slices = (
        slice(0, -win_lat),
        slice(-win_lat, -shift_lat),
        slice(-shift_lat, None),
    )
    lon_slices = (
        slice(0, -win_lon),
        slice(-win_lon, -shift_lon),
        slice(-shift_lon, None),
    )

    cnt = 0
    if ndim == 3:
        for pl in pl_slices:
            for lat in lat_slices:
                for lon in lon_slices:
                    img_mask[:, pl, lat, lon, :] = cnt
                    cnt += 1
        img_mask = img_mask[:, :, :, :Lon, :]
    elif ndim == 2:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, lat, lon, :] = cnt
                cnt += 1
        img_mask = img_mask[:, :, :Lon, :]

    mask_windows = window_partition(
        img_mask, window_size, ndim=ndim
    )  # n_lon, n_pl*n_lat, win_pl, win_lat, win_lon, 1 or n_lon, n_lat, win_lat, win_lon, 1
    if ndim == 3:
        win_total = win_pl * win_lat * win_lon
    elif ndim == 2:
        win_total = win_lat * win_lon
    mask_windows = mask_windows.view(
        mask_windows.shape[0], mask_windows.shape[1], win_total
    )
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask

def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = (
        padding_right
    ) = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return (
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_front,
        padding_back,
    )


def crop3d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): B, C, Pl, Lat, Lon
        resolution (tuple[int]): Pl, Lat, Lon
    """
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[
        :,
        :,
        padding_front : Pl - padding_back,
        padding_top : Lat - padding_bottom,
        padding_left : Lon - padding_right,
    ]

class PatchEmbed3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Image to Patch Embedding.

    Args:
        img_size (tuple[int]): Image size.
        patch_size (tuple[int]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim(int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        level, height, width = img_size
        l_patch_size, h_patch_size, w_patch_size = patch_size
        padding_left = (
            padding_right
        ) = padding_top = padding_bottom = padding_front = padding_back = 0

        l_remainder = level % l_patch_size
        h_remainder = height % l_patch_size
        w_remainder = width % w_patch_size

        if l_remainder:
            l_pad = l_patch_size - l_remainder
            padding_front = l_pad // 2
            padding_back = l_pad - padding_front
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = h_pad - padding_top
        if w_remainder:
            w_pad = w_patch_size - w_remainder
            padding_left = w_pad // 2
            padding_right = w_pad - padding_left

        self.pad = nn.ZeroPad3d(
            (
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
                padding_front,
                padding_back,
            )
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, L, H, W = x.shape
        x = self.pad(x)
        x = self.proj(x)
        if self.norm:
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x

class PatchRecovery3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[
            :,
            :,
            padding_front : Pl - padding_back,
            padding_top : Lat - padding_bottom,
            padding_left : Lon - padding_right,
        ]

class PatchEmbed2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D Image to Patch Embedding.

    Args:
        img_size (tuple[int]): Image size.
        patch_size (tuple[int]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim(int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        height, width = img_size
        h_patch_size, w_path_size = patch_size
        padding_left = padding_right = padding_top = padding_bottom = 0

        h_remainder = height % h_patch_size
        w_remainder = width % w_path_size

        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = int(h_pad - padding_top)

        if w_remainder:
            w_pad = w_path_size - w_remainder
            padding_left = w_pad // 2
            padding_right = int(w_pad - padding_left)

        self.pad = nn.ZeroPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.pad(x)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class PatchRecovery2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[
            :, :, padding_top : H - padding_bottom, padding_left : W - padding_right
        ]