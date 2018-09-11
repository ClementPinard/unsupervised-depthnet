from __future__ import division
import torch
from torch.autograd import Variable

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w)).type_as(depth)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w)).type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1,h,w)).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def compensate_pose(matrices, ref_matrix):
    check_sizes(matrices, 'matrices', 'BS34')
    check_sizes(ref_matrix, 'reference matrix', 'B34')
    translation_vectors = matrices[:,:,:,-1:] - ref_matrix[:,:,-1:].unsqueeze(1)
    inverse_rot = ref_matrix[:,:,:-1].transpose(1,2).unsqueeze(1)
    return inverse_rot @ torch.cat([matrices[:,:,:,:-1], translation_vectors], dim=-1)


def invert_mat(matrices):
    check_sizes(matrices, 'matrices', 'BS34')
    rot_matrices = matrices[:,:,:,:-1].transpose(2,3)
    translation_vectors = - rot_matrices @ matrices[:,:,:,-1:]
    return(torch.cat([rot_matrices, translation_vectors], dim=-1))


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    check_sizes(vec, 'rotation vector', 'BS6')
    translation = vec[:, :, :3].unsqueeze(-1)  # [B, S, 3, 1]
    rot = vec[:, :, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, S, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, S, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [B, S, 3, 4]
    return transform_mat


def pixel2cam(depth):
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    global pixel_coords
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    pixel_coords.type_as(depth)
    cam_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w)*depth.unsqueeze(1)
    return cam_coords.contiguous()


def cam2pixel(cam_coords, proj_c2p_rot=None, proj_c2p_tr=None):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
    X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
    Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
    Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b,h,w,2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, S, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, S, 3, 3]
    """
    B, S = angle.size()[:2]
    x, y, z = angle[:,:,0], angle[:,:,1], angle[:,:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=-1).view(B, S, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=-1).view(B, S, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=-1).view(B, S, 3, 3)
    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourth is then computed to have a norm of 1 -- size = [B, S, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, S, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=-1, keepdim=True)
    w, x, y, z = norm_quat[:,:,0], norm_quat[:,:,1], norm_quat[:,:,2], norm_quat[:,:,3]

    B, S = quat.size()[:2]

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, S, 3, 3)
    return rotMat


def inverse_warp(img, depth, pose_matrix, intrinsics, intrinsics_inv, rotation_mode='euler'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose_matrix, 'pose', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')

    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    rot_cam_to_src_pixel = intrinsics @ pose_matrix[:,:,:-1] @ intrinsics_inv  # [B, 3, 3]
    trans_cam_to_src_pixel = intrinsics @ pose_matrix[:,:,-1:]  # [B, 3, 1]

    src_pixel_coords = cam2pixel(cam_coords, rot_cam_to_src_pixel, trans_cam_to_src_pixel)  # [B,H,W,2]
    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords)

    return projected_img


def inverse_rotate(features, rot_matrix, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='border'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        features: the source image (where to sample pixels) -- [B, C, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(features, 'features', 'BCHW')
    check_sizes(rot_matrix, 'rotation matrix', 'B33')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')

    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, _, feature_height, feature_width = features.size()

    # construct a fake depth, with 1 everywhere
    depth = features[:,0].detach()*0 + 1

    cam_coords = pixel2cam(depth)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ rot_matrix @ intrinsics_inv  # [B, 3, 3]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel)  # [B,H,W,2]
    projected_img = torch.nn.functional.grid_sample(features, src_pixel_coords)

    return projected_img


def cam2depth(cam_coords, proj_c2p):
    """Transform depth values in the camera frame to the new frame. Notice the translation is done before rotation this time
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p: rotation and translation matrix of cameras -- [B, 3, 4]
    Returns:
        array of depth coordinates -- [B, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    pcoords = proj_c2p[:,:,:3].bmm(cam_coords_flat + proj_c2p[:,:,3:])  # [B, 3, H*W]
    Z = pcoords[:, 2].clamp(min=1e-3, max=100).view(b,h,w)
    return Z


def inverse_warp_depth(depth_to_warp, ref_depth, intrinsics, intrinsics_inv, rot=None, translation=None):
    """
    Inverse warp a source image to the target image plane.

    Args:
        depth_to_warp: the source depth (where to sample values) -- [B, H, W]
        ref_depth: target depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source depth warped to the target image plane
    """
    check_sizes(depth_to_warp, 'source depth', 'BHW')
    check_sizes(ref_depth, 'ref depth', 'BHW')
    # check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')

    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, img_height, img_width = depth_to_warp.size()

    cam_coords = pixel2cam(ref_depth, intrinsics_inv)  # [B,3,H,W]

    depth_to_warp_coords = pixel2cam(depth_to_warp, intrinsics_inv)  # [B,3,H,W]

    pose_mat = pose_vec2mat(rot, translation)  # [B,3,4]

    inverse_rot_mat = pose_vec2mat(-rot, None)  # [B,3,3]

    if translation is not None:
        inverse_translation = inverse_rot_mat.bmm(-translation.unsquee(-1))
        inverse_pose_mat = torch.cat([inverse_rot_mat, inverse_translation], dim=2)
    else:
        inverse_pose_mat = inverse_rot_mat

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel)  # [B,H,W,2]

    # change depths values of first depth map to match new pose, however depth is NOT resampled
    new_depth = cam2depth(depth_to_warp_coords, inverse_pose_mat)  # [B,H,W]

    projected_depth = torch.nn.functional.grid_sample(new_depth.unsqueeze(1), src_pixel_coords)[:,0]

    return projected_depth


def depth_forward_warp(depth, intrinsics, intrinsics_inv, pose, size_factor):
    b, h, w = depth.size()
    size = size_factor*(depth.contiguous()).view(b, -1)

    cam_coords = pixel2cam(depth, intrinsics_inv)
    pose_mat = pose_vec2mat(pose)
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, H*W, 3]
    proj_c2p_rot = proj_cam_to_src_pixel[:,:,:3]
    proj_c2p_tr = proj_cam_to_src_pixel[:,:,-1:]
    pcoords = proj_c2p_rot.bmm(cam_coords_flat) + proj_c2p_tr  # [B, 3, H*W]
    pcoords_transposed = pcoords.transpose(1,2)
    with_size = torch.cat((pcoords_transposed, size.view(b, -1, 1)), dim=-1)

    warped, index = warp_depth(with_size, (h, w))
    return warped, index


def occlusion_map(depth, intrinsics, intrinsics_inv, pose):
    b, h, w = depth.size()
    size = 1.5*(depth.contiguous()).view(b, -1)

    cam_coords = pixel2cam(depth, intrinsics_inv)
    pose_mat = pose_vec2mat(pose)
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, H*W, 3]
    proj_c2p_rot = proj_cam_to_src_pixel[:,:,:3]
    proj_c2p_tr = proj_cam_to_src_pixel[:,:,-1:]
    pcoords = proj_c2p_rot.bmm(cam_coords_flat) + proj_c2p_tr  # [B, 3, H*W]
    pcoords_transposed = pcoords.transpose(1,2)
    with_size = torch.cat((pcoords_transposed, size.view(b, -1, 1)), dim=-1)
    warped, index = warp_depth(with_size, (h, w))

    new_size = (warped.contiguous()).view(b, -1)
    new_cam_coords = pixel2cam(warped, intrinsics_inv)
    new_cam_coords_flat = new_cam_coords.view(b, 3, -1)

    inv_rot = pose_mat[:,:,:3].transpose(1,2)
    inv_tr = -inv_rot@pose_mat[:,:,-1:]
    new_pose_mat = torch.cat([inv_rot, inv_tr], dim=-1)
    new_proj_cam_to_src_pixel = intrinsics.bmm(new_pose_mat)
    new_proj_c2p_rot = new_proj_cam_to_src_pixel[:,:,:3]
    new_proj_c2p_tr = new_proj_cam_to_src_pixel[:,:,-1:]
    new_pcoords = new_proj_c2p_rot.bmm(new_cam_coords_flat) + new_proj_c2p_tr
    new_pcoords_transposed = new_pcoords.transpose(1,2)
    new_with_size = torch.cat((new_pcoords_transposed, new_size.view(b, -1, 1)), dim=-1)
    new_warped, new_index = warp_depth(new_with_size, (h, w))

    occulted = (new_warped.clamp(0,1e10) == 1e10)

    return occulted
