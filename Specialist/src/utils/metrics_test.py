import numpy as np
import torch


def to_cpu_tensor(*args):
    '''
    Input arbitrary number of array/tensors, each will be converted to CPU torch.Tensor
    '''
    out = []
    for tensor in args:
        if type(tensor) is np.ndarray:
            tensor = torch.Tensor(tensor)    
        if type(tensor) is torch.Tensor:
            tensor = tensor.cpu()
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
        (tn, fp), (fn, tp) = to_cpu_tensor(mat)
    return tp, tn, fp, fn

def csi(pred, targ, threshold):
    '''Critical Success Index. The larger the better.'''
    tp, tn, fp, fn = tfpn(pred, targ, threshold)
    if (tp + fn + fp) < 1e-7:
        return 0.
    return tp / (tp + fn + fp)

def ssim(y_pred, y):
    y, y_pred = to_cpu_tensor(y, y_pred)
    # to further ensure any of the input is not negative
    return torchmetrics.image.StructuralSimilarityIndexMeasure()(y_pred.unsqueeze(1), y.unsqueeze(1))

def psnr(y_pred, y):
    y, y_pred = to_cpu_tensor(y, y_pred)
    acc_score = 0
    B = y.shape[0]
    for i in range(B):
        acc_score += torchmetrics.image.PeakSignalNoiseRatio()(y_pred[i], y[i]) / B
    return acc_score

import lpips as lp

# GLOBAL_LPIPS_OBJ = None
# def lpips(y_pred, y, net='vgg'):
#     # convert the image range into [-1, 1], assuming the input range to be [0, 1]
#     upper = torch.max(torch.max(y_pred), torch.max(y))
#     y_pred, y = y_pred / upper, y / upper
#     y = (2 * y - 1)
#     y_pred = (2 * y_pred - 1)
#     global GLOBAL_LPIPS_OBJ
#     if GLOBAL_LPIPS_OBJ is None:
#         GLOBAL_LPIPS_OBJ = lp.LPIPS(net=net).to(y.device)
#     return GLOBAL_LPIPS_OBJ(y_pred, y).mean()

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
    
    stride = window // 2    
    pred = patches(pad(pred), window, stride) >= threshold
    gt = patches(pad(gt), window, stride) >= threshold
    pred_f = pred.sum(dim=[-1,-2]) / (pred.shape[-1] * pred.shape[-2])
    gt_f = gt.sum(dim=[-1,-2]) / (gt.shape[-1] * gt.shape[-2])
    score = 1 - ((pred_f - gt_f) ** 2).sum(dim=[-1]) / (pred_f ** 2 + gt_f ** 2).sum(dim=[-1])
    score[torch.isnan(score)] = 1.
    score = score.mean()
    return score

import torchist
# get patched historgrams
def get_patched_histograms(y, window, stride, bins, upper):
    y_p = patches(y, window, stride)
    counts = torch.zeros((len(y_p), bins))
    for i in range(len(y_p)):
        counts[i,:] = torchist.histogram(y_p[i].reshape(-1), bins=bins, low=0.0, upp=upper)
    return counts

def kl_div(gt, pred, dim=-1, epsilon=1e-5):
    # p log (p/q), where p is gt, q is pred.
    gt += epsilon 
    pred += epsilon            
    gt = gt / gt.sum(dim=dim, keepdim=True)
    pred = pred / pred.sum(dim=dim, keepdim=True)        
    return (gt * torch.log(gt / pred)).sum(dim=-1).mean()
    
def rhd(pred, targ, window=5, stride=2, bins=10):
    '''
    Regional Histogram Divergence (RHD)
    0 - inf, the KL divergence of two histogram distributions within a window
    '''
    upper = torch.max(torch.max(pred), torch.max(targ))
    hist_gt = get_patched_histograms(pad(targ), window, stride, bins, upper)
    hist_pred = get_patched_histograms(pad(pred), window, stride, bins, upper)
    return kl_div(hist_gt, hist_pred, dim=-1)

def fcl(pred, targ):
    if pred.dtype == torch.float16:
        pred = pred.float()
    # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
    fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
    fft_targ = torch.fft.fftn(targ, dim=[-1,-2], norm='ortho')
    conj_pred = torch.conj(fft_pred)
    numerator = (conj_pred*fft_targ).sum().real
    denominator = torch.sqrt(((fft_targ).abs()**2).sum()*((fft_pred).abs()**2).sum()) + 1E-7 
    return 1. - numerator/denominator

def fal(pred, targ):
    if pred.dtype == torch.float16:
        pred = pred.float()
    fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
    fft_targ = torch.fft.fftn(targ, dim=[-1,-2], norm='ortho')
    return torch.nn.MSELoss()(fft_pred.abs(), fft_targ.abs())
    
def cmpa_metrics_test(pred, targ, transform, exist_cmpa, variables, log_postfix, weighted=False, weight_dict=None):
    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            targ[i, 0] = pred[i, 0]

    pred = transform(pred)  # from normalized space to original space
    targ = transform(targ)
    pred, targ = pred[:, 0], targ[:, 0]
    loss_dict = {}
    with torch.no_grad():
        loss_dict[f"CSI_1_precp_{log_postfix}"] = csi(pred, targ, threshold=1.0)
        loss_dict[f"CSI_2_precp_{log_postfix}"] = csi(pred, targ, threshold=2.0)
        loss_dict[f"CSI_4_precp_{log_postfix}"] = csi(pred, targ, threshold=4.0)
        loss_dict[f"CSI_8_precp_{log_postfix}"] = csi(pred, targ, threshold=8.0)
        loss_dict[f"SSIM_precp_{log_postfix}"] = ssim(pred, targ)
        loss_dict[f"PSNR_precp_{log_postfix}"] = psnr(pred, targ)
        # loss_dict[f"LPIPS_precp_{log_postfix}"] = lpips(pred, targ)
        loss_dict[f"FSS_1_precp_{log_postfix}"] = fss(pred, targ, threshold=1)
        loss_dict[f"FSS_2_precp_{log_postfix}"] = fss(pred, targ, threshold=2)
        loss_dict[f"FSS_4_precp_{log_postfix}"] = fss(pred, targ, threshold=4)
        loss_dict[f"FSS_8_precp_{log_postfix}"] = fss(pred, targ, threshold=8)
        loss_dict[f"RHD_precp_{log_postfix}"] = rhd(pred, targ)
        loss_dict[f"FCL_precp_{log_postfix}"] = fcl(pred, targ)
        loss_dict[f"FAL_precp_{log_postfix}"] = fal(pred, targ)

    return loss_dict
