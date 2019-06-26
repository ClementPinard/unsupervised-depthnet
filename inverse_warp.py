from __future__ import division
import torch
import torch.nn.functional as F

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h, dtype=depth.dtype, device=depth.device).view(1, h, 1).expand(1,h,w)  # [1, H, W]
    j_range = torch.arange(0, w, dtype=depth.dtype, device=depth.device).view(1, 1, w).expand(1,h,w)  # [1, H, W]
    ones = torch.ones(1,h,w, dtype=depth.dtype, device=depth.device)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


@torch.jit.script
def compensate_pose(matrices, ref_matrix):
    # check_sizes(matrices, 'matrices', 'BS34')
    # check_sizes(ref_matrix, 'reference matrix', 'B34')
    translation_vectors = matrices[:,:,:,-1:] - ref_matrix[:,:,-1:].unsqueeze(1)
    inverse_rot = ref_matrix[:,:,:-1].transpose(1,2).unsqueeze(1)
    return inverse_rot @ torch.cat([matrices[:,:,:,:-1], translation_vectors], dim=-1)


@torch.jit.script
def invert_mat(matrices):
    # check_sizes(matrices, 'matrices', 'BS34')
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


@torch.jit.script
def cam2pixel(cam_coords):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    pcoords = cam_coords.view(b, 3, -1)  # [B, 3, H*W]

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b,h,w,2)


@torch.jit.script
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


@torch.jit.script
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


def inverse_warp(img, depth, pose_matrix, intrinsics, rotation_mode='euler'):
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
    intrinsics_inv = intrinsics.inverse()

    b, h, w = depth.shape
    batch_size, _, img_height, img_width = img.size()

    point_cloud = pixel2cam(depth)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    rot = intrinsics @ pose_matrix[:,:,:-1] @ intrinsics_inv  # [B, 3, 3]
    tr = intrinsics @ pose_matrix[:,:,-1:]

    transformed_points = rot @ point_cloud.view(b, 3, -1) + tr
    src_pixel_coords = cam2pixel(transformed_points.view(b, 3, h, w))  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode='border')

    with torch.no_grad():
        valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


def inverse_rotate(features, rot_matrix, intrinsics, rotation_mode='euler'):
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

    b, _, h, w = features.size()
    intrinsics_inv = intrinsics.inverse()

    # construct a fake depth, with 1 everywhere
    depth = features.new_ones([b, h, w])

    cam_coords = pixel2cam(depth)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    rot = intrinsics @ rot_matrix @ intrinsics_inv  # [B, 3, 3]
    transformed_points = rot @ cam_coords.view(b, 3, -1)
    src_pixel_coords = cam2pixel(transformed_points.view(b, 3, h, w))  # [B,H,W,2]
    projected_img = F.grid_sample(features, src_pixel_coords, padding_mode='border')

    return projected_img