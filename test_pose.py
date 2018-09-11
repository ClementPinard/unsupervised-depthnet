import torch
from torch.autograd import Variable

from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import PoseNet
from inverse_warp import pose_vec2mat, invert_mat, compensate_pose


parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path (for scale factor)")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")

parser.add_argument("--gt-type", default='KITTI', type=str, help="GroundTruth data type", choices=['npy', 'png', 'KITTI', 'stillbox'])
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)


@torch.no_grad()
def main():
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.gt_type == 'KITTI':
        from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework
    elif args.gt_type == 'stillbox':
        from stillbox_eval.pose_evaluation_utils import test_framework_stillbox as test_framework

    weights = torch.load(args.pretrained_posenet)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    pose_net = PoseNet(seq_length=seq_length).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']

        h,w,_ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

        imgs = [torch.from_numpy(np.transpose(img, (2,0,1))) for img in imgs]
        imgs = torch.stack(imgs).unsqueeze(0).to(device)
        imgs = 2*(imgs/255 - 0.5)

        poses = pose_net(imgs)

        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode)

        transform_matrices = invert_mat(inv_transform_matrices)

        # rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        # tr_vectors = rot_matrices @ inv_transform_matrices[:,:,-1:]

        # transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        # first_transform = transform_matrices[0]
        # final_poses = np.linalg.inv(first_transform[:,:3]) @ transform_matrices
        # final_poses[:,:,-1:] -= np.linalg.inv(first_transform[:,:3]) @ first_transform[:,-1:]

        final_poses = compensate_pose(transform_matrices, transform_matrices[:,0])[0].cpu().numpy()

        if args.output_dir is not None:
            predictions_array[j] = final_poses

        ATE, RE = compute_pose_error(sample['poses'][1:], final_poses[1:])
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)


def compute_pose_error(gt, pred):
    ATE = 0
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    for gt_pose, pred_pose in zip(gt, pred):
        ATE += np.linalg.norm(gt_pose[:,-1] - scale_factor * pred_pose[:,-1])

        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


if __name__ == '__main__':
    main()
