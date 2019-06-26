import torch
from torch.nn import functional as F
from inverse_warp import inverse_warp
import math
# from ssim import SSIM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ssim_mapper = SSIM(window_size=3)


def photometric_reconstruction_loss(imgs, tgt_indices, ref_indices,
                                    depth, pose, intrinsics,
                                    rotation_mode='euler', ssim_weight=0,
                                    upsample=False, occlusion=False):
    assert(pose.size(1) == imgs.size(1))
    b, _, h, w = depth.size()
    loss = torch.tensor(0, dtype=torch.float32, device=device)
    if b == 0:
        return loss, None, None
    batch_range = torch.arange(b, dtype=torch.int64, device=device)

    b, s, c, hi, wi = imgs.size()

    assert(hi >= h and wi >= w), "Depth size is greater than img size, which is probably not what you want"
    if upsample:
        imgs_scaled = imgs
        intrinsics_scaled = intrinsics
    else:
        downscale = hi/h
        imgs_scaled = F.interpolate(imgs, (c, h, w), mode='area')
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

    tgt_img_scaled = imgs_scaled[batch_range, tgt_indices]

    warped_results, diff, dssim, valid = [], [], [], []

    for i in range(s - 1):
        idx = ref_indices[:, i]
        current_pose = pose[batch_range, idx]
        ref_img = imgs[batch_range, idx]
        ref_img_warped, valid_points = inverse_warp(ref_img,
                                                    depth[:,0],
                                                    current_pose,
                                                    intrinsics_scaled,
                                                    rotation_mode,
                                                    occlusion)

        dssim_loss_map = (0.5*(1-ssim(tgt_img_scaled + 1, ref_img_warped + 1))).clamp(0,1) if ssim_weight > 0 else 0

        diff_map = tgt_img_scaled - ref_img_warped

        loss_map = ssim_weight * dssim_loss_map + (1-ssim_weight) * diff_map.abs()

        valid_loss_values = loss_map.masked_select(valid_points.unsqueeze(1))
        if valid_loss_values.numel() > 0:
            loss += valid_loss_values.mean()

        warped_results.append(ref_img_warped[0])
        dssim.append(dssim_loss_map[0])
        diff.append(diff_map[0])
        valid.append(valid_points[0])
    return loss, warped_results, diff, dssim, valid


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
    _2D_window = _1D_window @ (_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


window_size = 3
gaussian_img_kernel = create_gaussian_window(window_size, 3).float().to(device)


def grad_diffusion_loss(pred_disp, img=None, kappa=0.1):
    if type(pred_disp) not in [tuple, list]:
        pred_disp = [pred_disp]

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.shape
        if img is not None:
            with torch.no_grad():
                img_scaled = F.interpolate(img, (h, w), mode='area').norm(p=1, dim=1, keepdim=True)
                dx_i = img_scaled[:, :, 2:] - img_scaled[:, :, :-2]
                dy_i = img_scaled[:, :, :, 2:] - img_scaled[:, :, :, :-2]
                gx = torch.exp(-(dx_i.abs()/kappa)**2)
                gy = torch.exp(-(dy_i.abs()/kappa)**2)
        else:
            gx = gy = 1

        dx2 = scaled_disp[:,:, 2:] - 2 * scaled_disp[:,:,1:-1] + scaled_disp[:,:,:-2]
        dy2 = scaled_disp[:,:,:, 2:] - 2 * scaled_disp[:,:,:,1:-1] + scaled_disp[:,:,:,:-2]
        dx2 *= gx
        dy2 *= gy
        loss += (dx2.pow(2).mean() + dy2.pow(2).mean()) * weight
        weight /= 2
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
def compute_depth_errors(gt, pred, max_depth=80, crop=True):
    abs_diff, abs_rel, abs_log, a1, a2, a3 = 0,0,0,0,0,0
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
        valid = (current_gt > 0) & (current_gt < max_depth)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        abs_log += torch.mean(torch.abs(torch.log(valid_gt) - torch.log(valid_pred)))

    return [metric / b for metric in [abs_diff, abs_rel, abs_log, a1, a2, a3]]


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
            s = torch.stack([R[0,1]-R[1,0],R[1,2]-R[2,1],R[0,2]-R[2,0]]).norm(p=2)
            c = R.trace() - 1

            # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
            RE += torch.atan2(s,c)/seq_length

    return [ATE/batch_size, RE/batch_size]
