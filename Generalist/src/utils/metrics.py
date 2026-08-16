import numpy as np
import torch

def mse(pred, target, variables, exist_cmpa, weighted=False, weight_dict=None):
    """mean squared error

    Args:
        target: (B, V, H, W)
        pred: (B, V, H, W)
        variables: list of variable names
        exist_cmpa: (B)
    """
    # assert variables[0] == 'cmpa'

    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            target[i, 0] = pred[i, 0]

    error = (pred - target) ** 2  # （B, V, H, W）
    # error = error[exist_cmpa]
    
    loss_dict = {}
    for i, var in enumerate(variables):
        loss_dict[f"mse_{var}"] = error[:, i].mean()
    
    if weighted:
        weights = torch.Tensor([weight_dict[var] for var in variables]).to(device=error.device).view(1, -1, 1, 1)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(variables)).to(device=error.device).view(1, -1, 1, 1) / len(variables)
    
    # loss_dict["mse_aggregate"] = error.mean()

    loss_dict["w_mse_aggregate"] = (error * weights).sum(dim=1).mean()

    return loss_dict

def lat_weighted_mse(pred, target, variables, lat, weighted=False, weight_dict=None):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        target: (B, V, H, W)
        pred: (B, V, H, W)
        variables: list of variable names
        lat: H
    """

    error = (pred - target) ** 2  # （B, V, H, W）

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    for i, var in enumerate(variables):
        loss_dict[f"w_mse_{var}"] = (error[:, i] * w_lat).mean()
        
    if weighted:
        weights = torch.Tensor([weight_dict[var] for var in variables]).to(device=error.device).view(1, -1, 1, 1)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(variables)).to(device=error.device).view(1, -1, 1, 1) / len(variables)
    
    loss_dict["w_mse_aggregate"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()

    return loss_dict


def rmse(pred, target, transform, exist_cmpa, variables, log_postfix, weighted=False, weight_dict=None):
    """Root mean squared error

    Args:
        y: [B, V, H, W]
        target: [B, V, H, W]
        variables: list of variable names
        lat: H
    """

    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            target[i, 0] = pred[i, 0]

    normalized_error = (pred - target) ** 2

    if weighted:
        weights = torch.Tensor([weight_dict[var] for var in variables]).to(device=normalized_error.device)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(variables)).to(device=normalized_error.device)/ len(variables)

    # loss_dict["w_mse_aggregate"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()
    aggregate_normalized_rmse = torch.mean(
    torch.sqrt(torch.mean(normalized_error, dim=(-2, -1)))
    , dim=0)
    aggregate_normalized_rmse = (aggregate_normalized_rmse * weights).sum()

    pred = transform(pred)  # from normalized space to original space
    target = transform(target)

    error = (pred - target) ** 2  # [B, V, H, W]
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(variables):
            loss_dict[f"rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i], dim=(-2, -1)))
            )

    loss_dict[f"aggregate_normalized_rmse_{log_postfix}"] = aggregate_normalized_rmse
    # loss_dict[f"rmse_aggregate_{log_postfix}"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def lat_weighted_rmse(pred, target, transform, variables, lat, log_postfix, weighted=False, weight_dict=None):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        target: [B, V, H, W]
        variables: list of variable names
        lat: H
    """

    pred = transform(pred)  # from normalized space to original space
    target = transform(target)

    error = (pred - target) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(variables):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    loss_dict[f"w_rmse_aggregate_{log_postfix}"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def to_float_tensor(*args):
    '''
    Input arbitrary number of array/tensors, each will be converted to CPU torch.Tensor
    '''
    out = []
    for tensor in args:
        if type(tensor) is np.ndarray:
            tensor = torch.Tensor(tensor)    
        # if type(tensor) is torch.Tensor:
        #     tensor = tensor.cpu()
        tensor = tensor.float()
        out.append(tensor)
    # single value input: return single value output
    if len(out) == 1:
        return out[0]
    return out

import torchmetrics

def tfpn(y_pred, y, threshold):
    '''
    convert to cpu, and merge the first two dimensions
    '''
    with torch.no_grad():
        y = torch.where(y >= threshold, 1, 0)
        y_pred = torch.where(y_pred >= threshold, 1, 0)
        mat = torchmetrics.functional.confusion_matrix(y_pred, y, task='binary')
        (tn, fp), (fn, tp) = to_float_tensor(mat)
    return tp, tn, fp, fn

def csi(pred, targ, threshold):
    '''Critical Success Index. The larger the better.'''
    csi_set = []
    for i in range(pred.size(0)):
        tp, tn, fp, fn = tfpn(pred[i:i+1], targ[i:i+1], threshold)
        if (tp + fn + fp) < 1e-7:
            csi_set.append(0)
        else: csi_set.append(tp / (tp + fn + fp))
    return sum(csi_set)/len(csi_set)

def ssim(y_pred, y):
    y, y_pred = to_float_tensor(y, y_pred)
    # to further ensure any of the input is not negative
    return torchmetrics.image.StructuralSimilarityIndexMeasure().to(y.device)(y_pred.unsqueeze(1), y.unsqueeze(1))

def psnr(y_pred, y):
    y, y_pred = to_float_tensor(y, y_pred)
    psnr_score = 0
    B = y.shape[0]
    for i in range(B):
        psnr = torchmetrics.image.PeakSignalNoiseRatio().to(y.device)(y_pred[i], y[i]) / B
        if torch.isinf(psnr):
            continue  
        psnr_score += psnr
    return psnr_score

def patches(y, r, stride):
    p = y.unfold(-2, r, stride).unfold(-2, r, stride).reshape((-1, r, r))    
    return p

def pad(x):
    return torch.nn.functional.pad(x, (0, 0, 2, 2))

def fss(pred, gt, threshold=0.5, window=5):
    '''
    Fractional Skill Score (FSS) \\
    0 - 1, the higher the better.
    '''
    fss_set = []
    stride = window // 2    
    for i in range(pred.size(0)):
        pred_ = patches(pad(pred[i:i+1]), window, stride) >= threshold
        gt_ = patches(pad(gt[i:i+1]), window, stride) >= threshold
        pred_f = pred_.sum(dim=[-1,-2]) / (pred_.shape[-1] * pred_.shape[-2])
        gt_f = gt_.sum(dim=[-1,-2]) / (gt_.shape[-1] * gt_.shape[-2])
        denominator = (pred_f ** 2 + gt_f ** 2).sum(dim=[-1])
        numerator = ((pred_f - gt_f) ** 2).sum(dim=[-1])
        if denominator < 1e-7:
            score = 0
        else:  
            score = 1 - numerator / denominator
        fss_set.append(score)
    return sum(fss_set)/len(fss_set)


def fcl(pred, targ):
    # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
    if pred.dtype == torch.float16:
        pred = pred.float()
    fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
    fft_targ = torch.fft.fftn(targ, dim=[-1,-2], norm='ortho')
    conj_pred = torch.conj(fft_pred)
    numerator = (conj_pred*fft_targ).sum().real
    denominator = torch.sqrt(((fft_targ).abs()**2).sum()*((fft_pred).abs()**2).sum()) + 1E-7 
    return 1. - numerator/denominator

import torch.nn as nn

def fal(pred, targ):
    
    if pred.dtype == torch.float16:
        pred = pred.float()
    fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
    fft_targ = torch.fft.fftn(targ, dim=[-1,-2], norm='ortho')
    return nn.MSELoss()(fft_pred.abs(), fft_targ.abs())
    
def cmpa_metrics(pred, targ, transform, exist_cmpa, variables, log_postfix, weighted=False,weight_dict=None):
    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            targ[i, 0] = (pred[i, 0] + 0.001).clone()

    pred = transform(pred)  # from normalized space to original space
    targ = transform(targ)

    pred, targ = pred[:, 0].clone(), targ[:, 0].clone()
    loss_dict = {}
    with torch.no_grad():
        loss_dict[f"CSI_1_precp_{log_postfix}"] = csi(pred, targ, threshold=1.0)
        loss_dict[f"CSI_2_precp_{log_postfix}"] = csi(pred, targ, threshold=2.0)
        loss_dict[f"CSI_4_precp_{log_postfix}"] = csi(pred, targ, threshold=4.0)
        loss_dict[f"CSI_8_precp_{log_postfix}"] = csi(pred, targ, threshold=8.0)
        loss_dict[f"SSIM_precp_{log_postfix}"] = ssim(pred, targ)
        loss_dict[f"PSNR_precp_{log_postfix}"] = psnr(pred, targ)
        # loss_dict[f"{stage}/LPIPS_precp_{log_postfix}"] = lpips(pred, targ)
        loss_dict[f"FSS_1_precp_{log_postfix}"] = fss(pred, targ, threshold=1)
        loss_dict[f"FSS_2_precp_{log_postfix}"] = fss(pred, targ, threshold=2)
        loss_dict[f"FSS_4_precp_{log_postfix}"] = fss(pred, targ, threshold=4)
        loss_dict[f"FSS_8_precp_{log_postfix}"] = fss(pred, targ, threshold=8)
        # loss_dict[f"{stage}/RHD_precp_{log_postfix}"] = rhd(pred, targ)
        # loss_dict[f"FCL_precp_{log_postfix}"] = fcl(pred, targ)
        # loss_dict[f"FAL_precp_{log_postfix}"] = fal(pred, targ)

    return loss_dict

def cmpa_metrics_cf(pred, targ, transform, exist_cmpa, variables, log_postfix, weighted=False, weight_dict=None):
    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            targ[i, 0] = pred[i, 0] + 0.001
            
    loss_dict = {}
    # with torch.no_grad():
    #     norm_pred, norm_targ = pred[:, 0].clone(), targ[:, 0].clone()

    pred = transform(pred)  # from normalized space to original space
    targ = transform(targ)
    pred, targ = pred[:, 0], targ[:, 0]
    
    with torch.no_grad():
        loss_dict[f"CSI_5_precp_{log_postfix}"] = csi(pred, targ, threshold=5.0)
        loss_dict[f"CSI_10_precp_{log_postfix}"] = csi(pred, targ, threshold=10.0)
        loss_dict[f"FSS_5_precp_{log_postfix}"] = fss(pred, targ, threshold=5.0)
        loss_dict[f"FSS_10_precp_{log_postfix}"] = fss(pred, targ, threshold=10.0)

    return loss_dict

def lat_weighted_crps(pred: torch.Tensor, target: torch.Tensor, transform, variables, lat, log_postfix, weighted=False, weight_dict=None):
    assert len(pred.shape) == len(target.shape) + 1
    # pred: [B, N, V, H, W] because there are N ensemble members
    # target: [B, V, H, W]
    pred = transform(pred)
    target = transform(target)
    
    H, N = pred.shape[-2], pred.shape[1]
    
    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).to(dtype=pred.dtype, device=pred.device) # (H, )    
    
    def crps_var(pred_var: torch.Tensor, y_var: torch.Tensor):
        # pred_var: [B, N, H, W]
        # y: [B, H, W]
        # first term: prediction errors
        with torch.no_grad():
            error_term = torch.abs(pred_var - y_var.unsqueeze(1)) # [B, N, H, W]
            error_term = error_term * w_lat.view(1, 1, H, 1) # [B, N, H, W]
            error_term = torch.mean(error_term)
        
        # second term: ensemble spread
        with torch.no_grad():
            spread_term = torch.abs(pred_var.unsqueeze(2) - pred_var.unsqueeze(1)) # [B, N, N, H, W]
            spread_term = spread_term * w_lat.view(1, 1, 1, H, 1) # [B, N, N, H, W]
            spread_term = spread_term.mean(dim=(-2, -1)) # [B, N, N]
            spread_term = spread_term.sum(dim=(1, 2)) / (2 * N * (N - 1)) # [B]
            spread_term = spread_term.mean()
            
        return error_term - spread_term
    
    loss_dict = {}
    for i, var in enumerate(variables):
        loss_dict[f"w_crps_{var}_{log_postfix}"] = crps_var(pred[:, :, i], target[:, i])
        
    return loss_dict

def acc(pred, target, transform, exist_cmpa, vars, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            target[i, 0] = pred[i, 0]   
                     
    pred = transform(pred)
    target = transform(target)

    clim = clim.to(device=target.device)
    pred = pred - clim
    target = target - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_var = pred[:, i]
            target_var = target[:, i]
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.mean(
                torch.sum(pred_var * target_var, dim=(-1, -2)) / torch.sqrt(
                torch.sum(pred_var**2, dim=(-1, -2)) * torch.sum(target_var**2, dim=(-1, -2))
            )
            )

    return loss_dict

def rmse_center(pred, target, transform, exist_cmpa, variables, log_postfix, crop_rate, weighted=False, weight_dict=None):
    """Root mean squared error

    Args:
        y: [B, V, H, W]
        target: [B, V, H, W]
        variables: list of variable names
        lat: H
    """
    crop_lat, crop_lon = int(pred.size(2)*crop_rate//2), int(pred.size(3)*crop_rate//2)
    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            target[i, 0] = pred[i, 0]

    normalized_error = (pred - target) ** 2

    if weighted:
        weights = torch.Tensor([weight_dict[var] for var in variables]).to(device=normalized_error.device)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(variables)).to(device=normalized_error.device)/ len(variables)

    # loss_dict["w_mse_aggregate"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()
    aggregate_normalized_rmse = torch.mean(
    torch.sqrt(torch.mean(normalized_error, dim=(-2, -1)))
    , dim=0)
    aggregate_normalized_rmse = (aggregate_normalized_rmse * weights).sum()

    pred = transform(pred)  # from normalized space to original space
    target = transform(target)

    error = (pred - target) ** 2  # [B, V, H, W]
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(variables):
            loss_dict[f"rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i, crop_lat:-crop_lat, crop_lon:-crop_lon], dim=(-2, -1)))
            )

    loss_dict[f"aggregate_normalized_rmse_{log_postfix}"] = aggregate_normalized_rmse
    # loss_dict[f"rmse_aggregate_{log_postfix}"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def acc_center(pred, target, transform, exist_cmpa, vars, clim, log_postfix, crop_rate):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """
    crop_lat, crop_lon = int(pred.size(2)*crop_rate//2), int(pred.size(3)*crop_rate//2)
    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            target[i, 0] = pred[i, 0]   
                     
    pred = transform(pred)
    target = transform(target)

    clim = clim.to(device=target.device)
    pred = pred - clim
    target = target - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_var = pred[:, i, crop_lat:-crop_lat, crop_lon:-crop_lon]
            target_var = target[:, i, crop_lat:-crop_lat, crop_lon:-crop_lon]
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.mean(
                torch.sum(pred_var * target_var, dim=(-1, -2)) / torch.sqrt(
                torch.sum(pred_var**2, dim=(-1, -2)) * torch.sum(target_var**2, dim=(-1, -2))
            )
            )

    return loss_dict