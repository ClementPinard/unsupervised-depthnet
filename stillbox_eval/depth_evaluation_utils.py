import numpy as np
import json
from path import Path
from scipy.misc import imread
from tqdm import tqdm


class test_framework_stillbox(object):
    def __init__(self, root, test_files, seq_length=3, min_depth=1e-3, max_depth=80, step=1):
        self.root = root
        self.min_depth, self.max_depth = min_depth, max_depth
        self.gt_files, self.img_files, self.tgt_indices, self.poses, self.intrinsics = read_scene_data(self.root, test_files, seq_length, step)

    def __getitem__(self, i):
        depth = np.load(self.gt_files[i])
        return {'imgs': [imread(img).astype(np.float32) for img in self.img_files[i]],
                'tgt_index': self.tgt_indices[i],
                'path':self.img_files[i][0],
                'gt_depth': depth,
                'poses': self.poses[i],
                'mask': generate_mask(depth, self.min_depth, self.max_depth),
                'intrinsics': self.intrinsics
                }

    def __len__(self):
        return len(self.img_files)


def quat2mat(quat):
    w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = np.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], axis=1).reshape(quat.shape[0], 3, 3)
    return rotMat


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    sl = R.shape[0]
    R = R.reshape(sl, 3, 3)
    t = t.reshape(sl, 3, 1)
    filler = np.array([0,0,0,1]).reshape(1,1,4).repeat(sl, axis=0)
    basis = np.concatenate([R,t], axis=-1)
    final_matrix = np.concatenate([basis, filler], axis=1)
    return final_matrix


def get_poses(scene, indices):
    nominal_displacement = np.array(scene['speed']) * scene['time_step']
    sl = len(indices)
    if len(scene['orientation']) == 0:
        scene_quaternions = np.array([1,0,0,0]).reshape(1,4).repeat(sl, axis=0)
    else:
        scene_quaternions = np.array(scene['orientation'])[indices]
    t = np.array(indices).reshape(sl, 1) * nominal_displacement
    R = quat2mat(scene_quaternions).reshape(sl, 3, 3)
    matrices_seq = transform_from_rot_trans(R,t)
    matrices_seq = np.linalg.inv(matrices_seq[-1]) @ matrices_seq
    return matrices_seq[:,:3].astype(np.float32)


def read_scene_data(data_root, test_list, seq_length=3, step=1):
    data_root = Path(data_root)
    metadata_files = {}
    intrinsics = None
    for folder in data_root.dirs():
        with open(folder/'metadata.json', 'r') as f:
            metadata_files[str(folder.name)] = json.load(f)
        if intrinsics is None:
            args = metadata_files[str(folder.name)]['args']
            hfov = args['fov']
            w,h = args['resolution']
            f = w/(2*np.tan(np.pi*hfov/360))
            intrinsics = np.array([[f, 0, w/2],
                                   [0, f, h/2],
                                   [0, 0,   1]]).astype(np.float32)
    gt_files = []
    im_files = []
    poses = []
    tgt_indices = []
    shift_range = step * (np.arange(seq_length))

    print('getting test metadata ... ')
    for sample in tqdm(test_list):
        folder, file = sample.split('/')
        _, scene_index, index = file[:-4].split('_')  # filename is in the form 'RGB_XXXX_XX.jpg'
        index = int(index)
        scene = metadata_files[folder]['scenes'][int(scene_index)]
        scene_length = len(scene['imgs'])
        tgt_img_path = data_root/sample
        folder_path = data_root/folder
        if tgt_img_path.isfile():
            # if index is high enough, take only frames before. Otherwise, take only frames after.
            if index - shift_range[-1] > 0:
                ref_indices = index + shift_range - shift_range[-1]
                tgt_index = seq_length - 1
            elif index + shift_range[-1] < scene_length:
                ref_indices = index + shift_range
                tgt_index = 0
            else:
                raise
            tgt_indices.append(tgt_index)
            imgs_path = [folder_path/'{}'.format(scene['imgs'][ref_index]) for ref_index in ref_indices]

            gt_files.append(folder_path/'{}'.format(scene['depth'][index]))
            im_files.append(imgs_path)
            poses.append(get_poses(scene, ref_indices))
        else:
            print('{} missing'.format(tgt_img_path))

    return gt_files, im_files, tgt_indices, poses, intrinsics


def generate_mask(gt_depth, min_depth, max_depth):
    mask = np.logical_and(gt_depth > min_depth,
                          gt_depth < max_depth)
    # crop gt to exclude border values
    # if used on gt_size 100x100 produces a crop of [-95, -5, 5, 95]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.05 * gt_height, 0.95 * gt_height,
                     0.05 * gt_width,  0.95 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask
