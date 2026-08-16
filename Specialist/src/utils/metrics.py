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
        weights = torch.Tensor([weight_dict[var] for var in variables]).to(device=error.device)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(variables)).to(device=normalized_error.device) / len(variables)

    # loss_dict["w_mse_aggregate"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()
    aggregate_normalized_rmse = torch.mean(
    torch.sqrt(torch.mean(normalized_error, dim=(-2, -1)))
    , dim=0)
    aggregate_normalized_rmse = (aggregate_normalized_rmse * weights).sum()

    if transform:
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

import random
import torch.nn as nn
class RandomScheduling(nn.Module):
    def __init__(self, total_step, micro_batch, const_ratio=0.4):
        super(RandomScheduling, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        const_step = int(total_step*const_ratio)
        self.prob_thres = torch.linspace(1,0, int(total_step-const_step)).to(device)
        # dec = torch.linspace(1,0, int(total_step-const_step)).to(device)
        # if(const_ratio!=1):
        #     const = dec[-1] * torch.ones(const_step, device=device)
        # else:
        #     const = torch.zeros(int(total_step*const_ratio), device=device)
        # self.prob_thres = torch.cat((dec, const), dim=0)
        self.micro_batch = micro_batch
        self.step = 0
        self.out = 0

    def get_thres(self):
        if self.step % self.micro_batch == 0:
            prob = self.prob_thres[self.step//self.micro_batch] if self.step//self.micro_batch < len(self.prob_thres) else self.prob_thres[-1]
            self.out = 1 if random.random() > prob else 0
        self.step += 1
        return self.out

    def fcl(self, fft_pred, fft_truth):
        # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred*fft_truth).sum().real
        denominator = torch.sqrt(((fft_truth).abs()**2).sum()*((fft_pred).abs()**2).sum()) + 1E-7 
        return 1. - numerator/denominator

    def fal(self, fft_pred, fft_truth):
        return nn.MSELoss()(fft_pred.abs(), fft_truth.abs())

    def cos_loss(self, fft_pred, fft_truth):
        # Cosine Similarity Loss
        numerator = fft_pred.real*fft_truth.real + fft_pred.imag*fft_truth.imag
        denominator = fft_pred.abs()*fft_truth.abs() + 1E-7 
        return (1. - numerator/denominator).mean()

    def forward(self, pred, targ, exist_cmpa):

        for i in range(len(exist_cmpa)):
            if (exist_cmpa[i] == False):
                targ[i, 0] = pred[i, 0]

        pred, targ = pred[:, 0], targ[:, 0]
        if pred.dtype == torch.float16:
            pred = pred.float()
        fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
        fft_gt = torch.fft.fftn(targ, dim=[-1,-2], norm='ortho')
        # prob = 1 if random.random() > self.prob_thres[self.step] else 0
        prob = self.get_thres()

        _, H, W = pred.shape
        # weight = np.sqrt(H*W)
        loss = prob*self.fal(fft_pred, fft_gt) + (1-prob) * self.fcl(fft_pred, fft_gt)
        # loss = loss*weight
        # self.step += 1
        return loss

    def forward_fal_fcl(self, pred, targ, exist_cmpa):

        for i in range(len(exist_cmpa)):
            if (exist_cmpa[i] == False):
                targ[i, 0] = pred[i, 0]

        pred, targ = pred[:, 0], targ[:, 0]
        if pred.dtype == torch.float16:
            pred = pred.float()
        fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
        fft_gt = torch.fft.fftn(targ, dim=[-1,-2], norm='ortho')
        return self.fal(fft_pred, fft_gt), self.fcl(fft_pred, fft_gt)

class RandomScheduling_PhyAst(nn.Module):
    def __init__(self, total_step, micro_batch, const_ratio=0.4):
        super(RandomScheduling_PhyAst, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        const_step = int(total_step*const_ratio)
        self.prob_thres = torch.linspace(1,0, int(total_step-const_step)).to(device)
        # dec = torch.linspace(1,0, int(total_step-const_step)).to(device)
        # if(const_ratio!=1):
        #     const = dec[-1] * torch.ones(const_step, device=device)
        # else:
        #     const = torch.zeros(int(total_step*const_ratio), device=device)
        # self.prob_thres = torch.cat((dec, const), dim=0)
        self.micro_batch = micro_batch
        self.step = 0
        self.out = 0

    def get_thres(self):
        if self.step % self.micro_batch == 0:
            prob = self.prob_thres[self.step//self.micro_batch] if self.step//self.micro_batch < len(self.prob_thres) else self.prob_thres[-1]
            self.out = 1 if random.random() > prob else 0
        self.step += 1
        return self.out

    def fcl(self, fft_pred, fft_truth):
        # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred*fft_truth).sum().real
        denominator = torch.sqrt(((fft_truth).abs()**2).sum()*((fft_pred).abs()**2).sum()) + 1E-7 
        return 1. - numerator/denominator

    def fal(self, fft_pred, fft_truth):
        return nn.MSELoss()(fft_pred.abs(), fft_truth.abs())

    def cos_loss(self, fft_pred, fft_truth):
        # Cosine Similarity Loss
        numerator = fft_pred.real*fft_truth.real + fft_pred.imag*fft_truth.imag
        denominator = fft_pred.abs()*fft_truth.abs() + 1E-7 
        return (1. - numerator/denominator).mean()

    def forward(self, pred, targ, exist_cmpa):

        for i in range(len(exist_cmpa)):
            if (exist_cmpa[i] == False):
                targ[i, 0] = pred[i, 0]

        pred, targ = pred[:, 0], targ[:, 0]
        if pred.dtype == torch.float16:
            pred = pred.float()
        fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
        fft_gt = torch.fft.fftn(targ, dim=[-1,-2], norm='ortho')
        # prob = 1 if random.random() > self.prob_thres[self.step] else 0
        prob = self.get_thres()

        _, H, W = pred.shape
        # weight = np.sqrt(H*W)
        loss = prob*self.fal(fft_pred, fft_gt) + (1-prob) * self.fcl(fft_pred, fft_gt)
        # loss = loss*weight
        # self.step += 1
        return loss

    def forward_fal_fcl(self, pred, targ, exist_cmpa):

        for i in range(len(exist_cmpa)):
            if (exist_cmpa[i] == False):
                targ[i, 0] = pred[i, 0]

        pred, targ = pred[:, 0], targ[:, 0]
        if pred.dtype == torch.float16:
            pred = pred.float()
        fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
        fft_gt = torch.fft.fftn(targ, dim=[-1,-2], norm='ortho')
        return self.fal(fft_pred, fft_gt), self.fcl(fft_pred, fft_gt)
    
class ACSLoss(nn.Module):
    def __init__(self):
        super(ACSLoss, self).__init__()

    def acsloss(self, pred, targ):
        numerator = (pred*targ).sum()
        denominator = torch.sqrt((targ**2).sum()*(pred**2).sum()) + 1E-7 
        return 1. - numerator/denominator  

    def forward(self, pred, targ, exist_cmpa):

        new_targ = targ.clone()
        for i in range(len(exist_cmpa)):
            if (exist_cmpa[i] == False):
                new_targ[i, 0] = pred[i, 0]

        pred, new_targ = pred[:, 0], new_targ[:, 0]

        loss = self.acsloss(pred, new_targ)
        return loss

class ACSLoss_SampleWiseNorm(nn.Module):
    def __init__(self):
        super(ACSLoss_SampleWiseNorm, self).__init__()

    def acsloss(self, pred, targ):
        numerator = (pred*targ).sum()
        denominator = torch.sqrt((targ**2).sum()*(pred**2).sum()) + 1E-7 
        return 1. - numerator/denominator  

    def forward(self, pred, targ, exist_cmpa):

        new_targ = targ.clone()
        for i in range(len(exist_cmpa)):
            if (exist_cmpa[i] == False):
                new_targ[i, 0] = pred[i, 0]

        pred = pred[:, 0]
        new_targ = new_targ[:, 0]

        pred_mean = pred.mean(dim=[-1, -2], keepdim=True)
        targ_mean = new_targ.mean(dim=[-1, -2], keepdim=True)

        pred_centered = pred - pred_mean
        targ_centered = new_targ - targ_mean

        loss = self.acsloss(pred_centered, targ_centered)
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, targ, exist_cmpa):

        new_targ = targ.clone()
        for i in range(len(exist_cmpa)):
            if (exist_cmpa[i] == False):
                new_targ[i, 0] = pred[i, 0]

        pred, new_targ = pred[:, 0], new_targ[:, 0]

        loss = self.loss(pred, new_targ)
        return loss

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, targ, exist_cmpa):

        new_targ = targ.clone()
        for i in range(len(exist_cmpa)):
            if (exist_cmpa[i] == False):
                new_targ[i, 0] = pred[i, 0]

        pred, new_targ = pred[:, 0], new_targ[:, 0]

        loss = self.loss(pred, new_targ)
        return loss

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

# import lpips as lp

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

# import torchist
# # get patched historgrams
# def get_patched_histograms(y, window, stride, bins, upper):
#     y_p = patches(y, window, stride)
#     counts = torch.zeros((len(y_p), bins))
#     for i in range(len(y_p)):
#         counts[i,:] = torchist.histogram(y_p[i].reshape(-1), bins=bins, low=0.0, upp=upper)
#     return counts

# def kl_div(gt, pred, dim=-1, epsilon=1e-5):
#     # p log (p/q), where p is gt, q is pred.
#     gt += epsilon 
#     pred += epsilon            
#     gt = gt / gt.sum(dim=dim, keepdim=True)
#     pred = pred / pred.sum(dim=dim, keepdim=True)        
#     return (gt * torch.log(gt / pred)).sum(dim=-1).mean()
    
# def rhd(pred, targ, window=5, stride=2, bins=10):
#     '''
#     Regional Histogram Divergence (RHD)
#     0 - inf, the KL divergence of two histogram distributions within a window
#     '''
#     upper = torch.max(torch.max(pred), torch.max(targ))
#     hist_gt = get_patched_histograms(pad(targ), window, stride, bins, upper)
#     hist_pred = get_patched_histograms(pad(pred), window, stride, bins, upper)
#     return kl_div(hist_gt, hist_pred, dim=-1)

def cl(pred, targ):
    numerator = (pred*targ).sum()
    denominator = torch.sqrt((targ**2).sum()*(pred**2).sum()) + 1E-7 
    return 1. - numerator/denominator    

def corloss_log(pred, target, transform, exist_cmpa, variables, log_postfix, weighted=False, weight_dict=None):
    """Cosine Similarity Loss

    Args:
        y: [B, V, H, W]
        target: [B, V, H, W]
        variables: list of variable names
        lat: H
    """
    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            target[i, 0] = pred[i, 0]

    corloss = cl(pred, target)
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(variables):
            loss_dict[f"CorLoss_{var}_{log_postfix}"] = corloss

    return loss_dict
    
def cmpa_metrics(pred, targ, transform, exist_cmpa, variables, log_postfix, weighted=False, weight_dict=None):
    for i in range(len(exist_cmpa)):
        if (exist_cmpa[i] == False):
            # targ[i, 0] = pred[i, 0] + 0.001
            targ[i, 0] = pred[i, 0].clone()
            
    loss_dict = {}
    with torch.no_grad():
        norm_pred, norm_targ = pred[:, 0].clone(), targ[:, 0].clone()
        loss_dict[f"CorLoss_{log_postfix}"] = cl(norm_pred, norm_targ)

    pred = transform(pred)  # from normalized space to original space
    targ = transform(targ)
    pred, targ = pred[:, 0], targ[:, 0]
    
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

# def sc(R, VIWV, window=5):
#     '''
#     Fractional Skill Score (FSS) \\
#     0 - 1, the higher the better.
#     '''
    
#     stride = window // 2    
#     patch_R = patches(pad(R), window, stride).mean((1,2)) # (N, 5, 5)
#     patch_VIWV = patches(pad(VIWV), window, stride).mean((1,2)) # (N, 5, 5)

#     mu_R = patch_R.mean()
#     mu_VIWV = patch_VIWV.mean()

#     numerator = ((patch_R - mu_R) * (patch_VIWV - mu_VIWV)).sum()
#     denominator = torch.sqrt(((patch_R - mu_R)**2).sum() * ((patch_VIWV - mu_VIWV)**2).sum())

#     return numerator / (denominator + 1e-5)

def sc(R, VIWV, window=5):
    '''
    Fractional Skill Score (FSS) \\
    0 - 1, the higher the better.
    '''
    
    stride = window // 2    
    patch_R = torch.amax(patches(pad(R), window, stride), dim=(1, 2)) # (N, 5, 5)
    patch_VIWV = patches(pad(VIWV), window, stride).mean(dim=(1,2)) # (N, 5, 5)

    mu_R = patch_R.mean()
    mu_VIWV = patch_VIWV.mean()

    numerator = ((patch_R - mu_R) * (patch_VIWV - mu_VIWV)).sum()
    denominator = torch.sqrt(((patch_R - mu_R)**2).sum() * ((patch_VIWV - mu_VIWV)**2).sum())

    return numerator / (denominator + 1e-5)

def SpaCor_log(R, VIWV, transform, exist_cmpa, variables, log_postfix, weighted=False, weight_dict=None):
    """Spatial Correlation

    Args:
        R: [B, 1, H, W]
        VIWV: [B, 1, H, W]
        variables: list of variable names
    """
    Spatial_Correlation = sc(R, VIWV)
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(variables):
            loss_dict[f"{var}_{log_postfix}"] = Spatial_Correlation

    return loss_dict