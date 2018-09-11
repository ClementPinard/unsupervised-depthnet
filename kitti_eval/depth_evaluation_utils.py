# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
# import pandas as pd
from collections import Counter
from path import Path
from scipy.misc import imread
from tqdm import tqdm


class test_framework_KITTI(object):
    def __init__(self, root, test_files, seq_length=3, min_depth=1e-3, max_depth=100, step=1):
        self.root = root
        self.min_depth, self.max_depth = min_depth, max_depth
        self.calib_dirs, self.gt_files, self.img_files, self.tgt_indices, self.poses, self.cams = read_scene_data(self.root, test_files, seq_length, step)

    def __getitem__(self, i):
        tgt = imread(self.img_files[i][0]).astype(np.float32)
        depth = generate_depth_map(self.calib_dirs[i], self.gt_files[i], tgt.shape[:2], self.cams[i])
        intrinsics = read_calib_file(self.calib_dirs[i]/'calib_cam_to_cam.txt')['P_rect_02'].reshape(3,4)[:,:3]

        return {'imgs': [imread(img).astype(np.float32) for img in self.img_files[i]],
                'tgt_index': self.tgt_indices[i],
                'path': self.img_files[i][0],
                'gt_depth': depth,
                'poses': self.poses[i],
                'mask': generate_mask(depth, self.min_depth, self.max_depth),
                'intrinsics': intrinsics
                }

    def __len__(self):
        return len(self.img_files)


###############################################################################
#  EIGEN

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def pose_from_oxts_packet(metadata):

    lat, lon, alt, roll, pitch, yaw, *_ = metadata
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    Taken from https://github.com/utiasSTARS/pykitti
    """

    er = 6378137.  # earth radius (approx.) in meters
    # Use a Mercator projection to get the translation vector
    ty = lat * np.pi * er / 180.

    scale = np.cos(lat * np.pi / 180.)
    tx = scale * lon * np.pi * er / 180.
    # ty = scale * er * \
    #     np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz]).reshape(-1,1)

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))
    return transform_from_rot_trans(R, t)


def get_pose_matrices(oxts_root, imu2cam, indices):
    matrices = []
    for index in indices:
        oxts_data = np.genfromtxt(oxts_root/'data'/'{:010d}.txt'.format(index))
        pose = pose_from_oxts_packet(oxts_data)
        odo_pose = pose @ np.linalg.inv(imu2cam)
        matrices.append(odo_pose)
    matrices_seq = np.stack(matrices)
    matrices_seq[:,:3,-1] -= matrices_seq[-1,:3,-1]
    matrices_seq = np.linalg.inv(matrices_seq[-1]) @ matrices_seq
    return matrices_seq[:,:3].astype(np.float32)


def read_scene_data(data_root, test_list, seq_length=3, step=1):
    data_root = Path(data_root)
    gt_files = []
    calib_dirs = []
    im_files = []
    cams = []
    poses = []
    tgt_indices = []
    shift_range = step * (np.arange(seq_length))

    print('getting test metadata ... ')
    for sample in tqdm(test_list):
        tgt_img_path = data_root/sample
        date, scene, cam_id, _, index = sample[:-4].split('/')
        index = int(index)

        imu2velo = read_calib_file(data_root/date/'calib_imu_to_velo.txt')
        velo2cam = read_calib_file(data_root/date/'calib_velo_to_cam.txt')
        cam2cam = read_calib_file(data_root/date/'calib_cam_to_cam.txt')

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

        scene_length = len(tgt_img_path.parent.files('*.png'))

        # if index is high enough, take only frames before. Otherwise, take only frames after.
        if index - shift_range[-1] > 0:
            ref_indices = index + shift_range - shift_range[-1]
            tgt_index = seq_length - 1
        elif index + shift_range[-1] < scene_length:
            ref_indices = index + shift_range
            tgt_index = 0
        else:
            raise

        imgs_path = [tgt_img_path.dirname()/'{:010d}.png'.format(i) for i in ref_indices]
        vel_path = data_root/date/scene/'velodyne_points'/'data'/'{:010d}.bin'.format(index)

        if tgt_img_path.isfile():
            gt_files.append(vel_path)
            calib_dirs.append(data_root/date)
            im_files.append(imgs_path)
            cams.append(int(cam_id[-2:]))
            poses.append(get_pose_matrices(data_root/date/scene/'oxts', imu2cam, ref_indices))
            tgt_indices.append(tgt_index)
        else:
            print('{} missing'.format(tgt_img_path))

    return calib_dirs, gt_files, im_files, tgt_indices, poses, cams


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:,3] = 1
    return points


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.float32(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def get_focal_length_baseline(calib_dir, cam=2):
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    P2_rect = cam2cam['P_rect_02'].reshape(3,4)
    P3_rect = cam2cam['P_rect_03'].reshape(3,4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0,3] / -P2_rect[0,0]
    b3 = P3_rect[0,3] / -P3_rect[0,0]
    baseline = b3-b2

    if cam == 2:
        focal_length = P2_rect[0,0]
    elif cam == 3:
        focal_length = P3_rect[0,0]

    return focal_length, baseline


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2):
    # load calibration files
    cam2cam = read_calib_file(calib_dir/'calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir/'calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,-1:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    return depth


def generate_mask(gt_depth, min_depth, max_depth):
    mask = np.logical_and(gt_depth > min_depth,
                          gt_depth < max_depth)
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask
