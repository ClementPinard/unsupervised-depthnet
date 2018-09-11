import numpy as np
import torch.utils.data as data
from imageio import imread
import random
import json
from path import Path


def load_as_float(path):
    return imread(path).astype(np.float32)


def quat2mat(quat):
    w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = np.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], axis=1).reshape(quat.shape[0], 3, 3)
    return rotMat


class ShiftedSequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        (optional) root/scene_1/shifts.json
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, with_pose_gt=False, with_depth_gt=False,
                 sequence_length=3, target_displacement=0.02, transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.target_displacement = target_displacement
        self.max_shift = 10
        self.adjust = False
        self.with_pose_gt = with_pose_gt
        self.with_depth_gt = with_depth_gt
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        img_sequences = []
        poses_sequences = []
        demi_length = sequence_length//2
        for scene in self.scenes:
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue

            shifts_file = scene/'shifts.json'
            if shifts_file.isfile():
                with open(shifts_file, 'r') as f:
                    shifts = json.load(f)
            else:
                prior_shifts = list(range(-demi_length, 0))
                post_shifts = list(range(1, sequence_length - demi_length))
                shifts = [[prior_shifts[:], post_shifts[:]] for i in imgs]

            if self.with_pose_gt:
                pose_file = scene/'poses.txt'
                assert pose_file.isfile(), "cannot find ground truth pose file {}".format(pose_file)
                poses = np.loadtxt(pose_file).astype(np.float32).reshape(-1,3,4)
                poses_sequences.append(poses)
            img_sequences.append(imgs)
            sequence_index = len(img_sequences) - 1
            intrinsics = np.loadtxt(scene/'cam.txt').astype(np.float32).reshape(3,3)
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics,
                          'tgt': i,
                          'prior_shifts': shifts[i][0],
                          'post_shifts': shifts[i][1],
                          'sequence_index': sequence_index}
                if self.with_depth_gt:
                    depth = imgs[i].stripext() + '.npy'
                    assert depth.isfile(), "cannot find ground truth depth map {}".format(depth)
                    sample['depth'] = depth
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
        self.img_sequences = img_sequences
        if self.with_pose_gt:
            self.poses_sequences = poses_sequences

    def __getitem__(self, index):
        sample = self.samples[index]
        preprocessed_sample = {}
        imgs = self.img_sequences[sample['sequence_index']]
        tgt_index = sample['tgt']
        tgt_img = load_as_float(imgs[tgt_index])
        if self.with_depth_gt:
            tgt_depth = np.load(sample['depth'])
            preprocessed_sample['depth'] = tgt_depth

        try:
            prior_imgs = [load_as_float(imgs[tgt_index + i]) for i in sample['prior_shifts']]
            post_imgs = [load_as_float(imgs[tgt_index + i]) for i in sample['post_shifts']]
            imgs = prior_imgs + [tgt_img] + post_imgs
            if self.with_pose_gt:
                poses = self.poses_sequences[sample['sequence_index']]
                tgt_pose = poses[tgt_index]
                prior_poses = [poses[tgt_index + i] for i in sample['prior_shifts']]
                post_poses = [poses[tgt_index + i] for i in sample['post_shifts']]
                pose_sequence = np.stack(prior_poses + [tgt_pose] + post_poses)
                # neutral pose is defined to be last frame
                pose_sequence[:,:,-1] -= pose_sequence[-1,:,-1]
                compensated_poses = np.linalg.inv(pose_sequence[-1,:,:3]) @ pose_sequence
                preprocessed_sample['pose'] = compensated_poses
        except Exception as e:
            print(index, sample['tgt'], sample['prior_shifts'], sample['post_shifts'], len(imgs))
            raise e
        if self.transform is not None:
            imgs, intrinsics = self.transform(imgs, sample['intrinsics'])
        else:
            intrinsics = sample['intrinsics']
        preprocessed_sample['imgs'] = imgs
        preprocessed_sample['intrinsics'] = intrinsics
        preprocessed_sample['intrinsics_inv'] = np.linalg.inv(intrinsics)
        if self.adjust:
            preprocessed_sample['index'] = index
        return preprocessed_sample

    def reset_shifts(self, index, prior_ratio, post_ratio):
        sample = self.samples[index]
        assert(len(sample['prior_shifts']) == len(prior_ratio))
        assert(len(sample['post_shifts']) == len(post_ratio))
        imgs = self.img_sequences[sample['sequence_index']]
        tgt_index = sample['tgt']

        for j, r in enumerate(prior_ratio[::-1]):

            shift_index = len(prior_ratio) - 1 - j
            old_shift = sample['prior_shifts'][shift_index]
            new_shift = old_shift * r
            assert(new_shift < 0), "shift must be negative: {:.3f}, {}, {:.3f}".format(new_shift, old_shift, r)
            new_shift = round(new_shift)
            ''' Here is how bounds work for prior shifts:
            prior shifts must be negative in a strict ascending order in the original list
            max_shift (in magnitude) is either tgt (to keep index inside list) or self.max_shift
            Let's say you have 2 anterior shifts, which means seq_length is 5
            1st shift can be -max_shift but cannot be 0 as it would mean that 2nd would not be higher than 1st and above 0
            2nd shift cannot be -max_shift as 1st shift would have to be less than -max_shift - 1.
            More generally, shift must be clipped within -max_shift + its index and upper shift - 1
            Note that priority is given for shifts closer to tgt_index, they are free to choose the value they want, at the risk of
            constraining outside shifts to one only valid value
            '''

            max_shift = min(tgt_index, self.max_shift)

            lower_bound = -max_shift + shift_index
            upper_bound = -1 if shift_index == len(prior_ratio) - 1 else sample['prior_shifts'][shift_index + 1] - 1

            sample['prior_shifts'][shift_index] = int(np.clip(new_shift, lower_bound, upper_bound))

        for j, r in enumerate(post_ratio):
            shift_index = j
            old_shift = sample['post_shifts'][shift_index]
            new_shift = old_shift * r
            assert(new_shift > 0), "shift must be positive: {:.3f}, {}, {}".format(new_shift, old_shift, r)
            new_shift = round(new_shift)
            '''For posterior shifts :
            must be postive in a strict descending order
            max_shift is either len(imgs) - tgt or self.max_shift
            shift must be clipped within upper shift + 1 and max_shift - seq_length + its index
            '''

            max_shift = min(len(imgs) - tgt_index - 1, self.max_shift)

            lower_bound = 1 if shift_index == 0 else sample['post_shifts'][shift_index - 1] + 1
            upper_bound = max_shift + shift_index - len(post_ratio) + 1

            sample['post_shifts'][shift_index] = int(np.clip(new_shift, lower_bound, upper_bound))

    def get_shifts(self, index):
        sample = self.samples[index]
        prior = sample['prior_shifts']
        post = sample['post_shifts']
        return prior + post

    def __len__(self):
        return len(self.samples)


class StillBox(ShiftedSequenceFolder):
    def crawl_folders(self, sequence_length):
        import json
        sequence_set = []
        img_sequences = []
        poses_sequences = []
        demi_length = sequence_length//2
        for folder in self.scenes:
            with open(folder/'metadata.json', 'r') as f:
                metadata = json.load(f)
            args = metadata['args']
            hfov = args['fov']
            w,h = args['resolution']
            f = w/(2*np.tan(np.pi*hfov/360))
            intrinsics = np.array([[f, 0, w/2],
                                   [0, f, h/2],
                                   [0, 0,   1]]).astype(np.float32)
            for scene in metadata['scenes']:
                imgs = [folder/i for i in scene['imgs']]
                if self.with_depth_gt:
                    depth = [folder/i for i in scene['depth']]
                if len(imgs) < sequence_length:
                    continue

                prior_shifts = list(range(-demi_length, 0))
                post_shifts = list(range(1, sequence_length - demi_length))
                shifts = [[prior_shifts[:], post_shifts[:]] for i in imgs]

                if self.with_pose_gt:
                    sl = len(scene['imgs'])
                    nominal_displacement = np.float32(scene['speed']) * scene['time_step']
                    if len(scene['orientation']) == 0:
                        scene_quaternions = np.float32([1,0,0,0]).reshape(1,4).repeat(sl, axis=0)
                    else:
                        scene_quaternions = np.float32(scene['orientation'])
                    scene_positions = np.arange(sl).astype(np.float32).reshape(sl, 1) * nominal_displacement
                    orientation_matrices = quat2mat(scene_quaternions).reshape(sl, 3, 3)
                    pose_matrices = np.concatenate((orientation_matrices, scene_positions.reshape(sl, 3, 1)), axis=2)
                    poses_sequences.append(pose_matrices)

                img_sequences.append(imgs)
                sequence_index = len(img_sequences) - 1

                for i in range(demi_length, len(imgs)-demi_length):
                    sample = {'intrinsics': intrinsics,
                              'tgt': i,
                              'prior_shifts': shifts[i][0],
                              'post_shifts': shifts[i][1],
                              'sequence_index': sequence_index}
                    if self.with_depth_gt:
                        sample['depth'] = depth[i]

                    sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
        self.img_sequences = img_sequences
        if self.with_pose_gt:
            self.poses_sequences = poses_sequences