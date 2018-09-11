import torch
from torch.nn import functional as F
from inverse_warp import inverse_warp
import numpy as np
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def photometric_reconstruction_loss(imgs, tgt_indices, ref_indices, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', ssim_weight=0):
    assert(pose.size(1) == imgs.size(1))
    b, _, h, w = depth.size()
    batch_range = torch.arange(b).long().to(device)

    b, s, _, hi, wi = imgs.size()
    downscale = hi/h
    imgs_scaled = F.interpolate(imgs, (3, h, w), mode='trilinear', align_corners=False)

    intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
    intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
    tgt_img_scaled = imgs_scaled[batch_range, tgt_indices]

    loss = 0
    warped_results, diff = [], []

    for i in range(s - 1):
        idx = ref_indices[:, i]
        current_pose = pose[batch_range, idx]
        ref_img = imgs[batch_range, idx]
        ref_img_warped = inverse_warp(ref_img,
                                      depth[:,0],
                                      current_pose,
                                      intrinsics_scaled,
                                      intrinsics_scaled_inv,
                                      rotation_mode)

        warped_results.append(ref_img_warped)

        out_of_bounds = (ref_img_warped == 0).prod(1, keepdim=True, dtype=torch.uint8)

        ssim_loss_map = (0.5*(1-ssim(tgt_img_scaled, ref_img_warped))).clamp(0,1) if ssim_weight > 0 else 0

        diff_map = (tgt_img_scaled - ref_img_warped).abs()
        diff.append(diff_map * (1 - out_of_bounds.type_as(ref_img_warped)))

        loss_map = ssim_weight * ssim_loss_map + (1-ssim_weight) * diff_map

        valid_loss_values = loss_map.masked_select(~out_of_bounds)
        if valid_loss_values.numel() > 0:
            loss += valid_loss_values.abs().mean()

    return loss, diff, warped_results


def smooth_loss(pred_disp):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_disp) not in [tuple, list]:
        pred_disp = [pred_disp]

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:

        dx, dy = gradient(scaled_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight / scaled_disp.mean()
        weight /= 4
    return loss


grad_kernel = torch.FloatTensor([[ 1, 2, 1],
                                 [ 0, 0, 0],
                                 [-1,-2,-1]]).view(1,1,3,3).to(device)/4
grad_img_kernel = grad_kernel.expand(3,1,3,3).contiguous()
lapl_kernel = torch.FloatTensor([[-1,-2,-1],
                                 [-2,12,-2],
                                 [-1,-2,-1]]).view(1,1,3,3).to(device)/12


def create_gaussian_window(window_size, channel):
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window@(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


window_size = 11
gaussian_img_kernel = create_gaussian_window(window_size, 3).float().to(device)


def texture_aware_smooth_loss(pred_map, image=None):
    global grad_img_kernel, lapl_kernel

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.
    eps = 0.1

    for scaled_map in pred_map:
        if image is not None:
            b, _, h, w = scaled_map.size()
            scaled_image = F.adaptive_avg_pool2d(image.detach(), (h, w))
            grad_y = F.conv2d(scaled_image, grad_img_kernel, groups=3)
            grad_x = F.conv2d(scaled_image, grad_img_kernel.transpose(2,3).contiguous(), groups=3)
            textureness = (grad_x.abs() + grad_y.abs()).sum(dim=1, keepdim=True) + eps
        else:
            textureness = 1

        disp_lapl = F.conv2d(scaled_map, lapl_kernel.type_as(scaled_map))
        loss_map = disp_lapl / textureness

        loss += loss_map.abs().mean()*weight / scaled_map.detach().mean()
        weight /= 4
    return loss


def ssim(img1, img2):
    params = {'weight': gaussian_img_kernel, 'groups':3, 'padding':window_size//2}
    mu1 = F.conv2d(img1, **params)
    mu2 = F.conv2d(img2, **params)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, **params) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, **params) - mu2_sq
    sigma12 = F.conv2d(img1*img2, **params) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map


@torch.no_grad()
def compute_depth_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    b, h, w = gt.size()
    if pred.size(1) != h:
        pred_upscaled = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)[:,0]
    else:
        pred_upscaled = pred[0:,]

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred_upscaled):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / b for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


@torch.no_grad()
def compute_pose_error(gt, pred):
    ATE = 0
    RE = 0
    batch_size, seq_length = gt.size()[:2]
    for gt_pose_seq, pred_pose_seq in zip(gt, pred):
        scale_factor = (gt_pose_seq[:,:,-1] * pred_pose_seq[:,:,-1]).sum()/(pred_pose_seq[:,:,-1] ** 2).sum()
        for gt_pose, pred_pose in zip(gt_pose_seq, pred_pose_seq):
            ATE += ((gt_pose[:,-1] - scale_factor * pred_pose[:,-1]).norm(p=2))/seq_length

            # Residual matrix to which we compute angle's sin and cos
            R = gt_pose[:,:3] @ pred_pose[:,:3].inverse()
            s = np.linalg.norm([R[0,1]-R[1,0],
                                R[1,2]-R[2,1],
                                R[0,2]-R[2,0]])
            c = np.trace(R) - 1

            # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
            RE += np.arctan2(s,c)/seq_length

    return [ATE/batch_size, RE/batch_size]
