import torch
import torch.nn.functional as F
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import imageio

from models import DepthNet, PoseNet
from inverse_warp import pose_vec2mat, compensate_pose, invert_mat, inverse_rotate
from utils import tensor2array


parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-depthnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path (for scale factor)")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80, type=float)
parser.add_argument("--stabilize-from-GT", action='store_true')
parser.add_argument("--nominal-displacement", type=float, default=0.3)
parser.add_argument("--output-dir", default='.', type=str, help="Output directory for saving")
parser.add_argument("--log-best-worst", action='store_true', help="if selected, will log depthNet outputs")
parser.add_argument("--save-output", action='store_true', help="if selected, will save all predictions in a big 3D numpy file")

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")

parser.add_argument("--gt-type", default='KITTI', type=str, help="GroundTruth data type", choices=['npy', 'png', 'KITTI', 'stillbox'])
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)


target_mean_depthnet_output = 50
best_error = np.inf
worst_error = 0


def select_best_map(maps, target_mean):
    unraveled_maps = maps.view(maps.size(0), -1)
    means = unraveled_maps.mean(1)  # this should be a 1D tensor
    best_index = torch.min((means-target_mean).abs(), 0)[1].item()
    best_map = maps[best_index,0]
    return best_map, best_index


def log_result(pred_depth, GT, input_batch, selected_index, folder, prefix):
    def save(path, to_save):
        to_save = (255*to_save.transpose(1,2,0)).astype(np.uint8)
        imageio.imsave(path, to_save)
    pred_to_save = tensor2array(pred_depth, max_value=100)
    gt_to_save = tensor2array(torch.from_numpy(GT), max_value=100)

    prefix = folder/prefix
    save('{}_depth_pred.jpg'.format(prefix), pred_to_save)
    save('{}_depth_gt.jpg'.format(prefix), gt_to_save)
    disp_to_save = tensor2array(1/pred_depth, max_value=None, colormap='magma')
    gt_disp = np.zeros_like(GT)
    valid_depth = GT > 0
    gt_disp[valid_depth] = 1/GT[valid_depth]

    gt_disp_to_save = tensor2array(torch.from_numpy(gt_disp), max_value=None, colormap='magma')
    save('{}_disp_pred.jpg'.format(prefix), disp_to_save)
    save('{}_disp_gt.jpg'.format(prefix), gt_disp_to_save)
    to_save = tensor2array(input_batch.cpu().data[selected_index,:3])
    save('{}_input0.jpg'.format(prefix), to_save)
    to_save = tensor2array(input_batch.cpu()[selected_index,3:])
    save('{}_input1.jpg'.format(prefix), to_save)
    for i, batch_elem in enumerate(input_batch.cpu().data):
        to_save = tensor2array(batch_elem[:3])
        save('{}_batch_{}_0.jpg'.format(prefix, i), to_save)
        to_save = tensor2array(batch_elem[3:])
        save('{}_batch_{}_1.jpg'.format(prefix, i), to_save)


@torch.no_grad()
def main():
    global best_error, worst_error
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parser.parse_args()
    if args.gt_type == 'KITTI':
        from kitti_eval.depth_evaluation_utils import test_framework_KITTI as test_framework
    elif args.gt_type == 'stillbox':
        from stillbox_eval.depth_evaluation_utils import test_framework_stillbox as test_framework

    weights = torch.load(args.pretrained_depthnet)
    depthnet_params = {"depth_activation":"elu",
                       "batch_norm":"bn" in weights.keys() and weights['bn']}
    if not args.no_resize:
        depthnet_params['input_size'] = (args.img_height, args.img_width)
        depthnet_params['upscale'] = True

    depth_net = DepthNet(**depthnet_params).to(device)
    depth_net.load_state_dict(weights['state_dict'])
    depth_net.eval()

    if args.pretrained_posenet is None:
        args.stabilize_from_GT = True
        print('no PoseNet specified, stab will be done from ground truth')
        seq_length = 5
    else:
        weights = torch.load(args.pretrained_posenet)
        seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
        posenet_params = {'seq_length':seq_length}
        if not args.no_resize:
            posenet_params['input_size'] = (args.img_eight, args.img_width)

        pose_net = PoseNet(**posenet_params).to(device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
    else:
        test_files = [file.relpathto(dataset_dir) for file in sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])]

    framework = test_framework(dataset_dir, test_files, seq_length, args.min_depth, args.max_depth)

    print('{} files to test'.format(len(test_files)))
    errors = np.zeros((9, len(test_files)), np.float32)

    args.output_dir = Path(args.output_dir)
    args.output_dir.makedirs_p()

    for j, sample in enumerate(tqdm(framework)):
        intrinsics = torch.from_numpy(sample['intrinsics']).unsqueeze(0).to(device)
        imgs = sample['imgs']
        imgs = [torch.from_numpy(np.transpose(img, (2,0,1))) for img in imgs]
        imgs = torch.stack(imgs).unsqueeze(0).to(device)
        imgs = 2*(imgs/255 - 0.5)

        tgt_img = imgs[:,sample['tgt_index']]

        # Construct a batch of all possible stabilized pairs, with PoseNet or with GT orientation, will take the output closest to target mean depth
        if args.stabilize_from_GT:
            poses_GT = torch.from_numpy(sample['poses']).unsqueeze(0).to(device)
            inv_poses_GT = invert_mat(poses_GT)
            tgt_pose = inv_poses_GT[:,sample['tgt_index']]
            inv_transform_matrices_tgt = compensate_pose(inv_poses_GT, tgt_pose)
        else:
            poses = pose_net(imgs)
            inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode)

            tgt_pose = inv_transform_matrices[:,sample['tgt_index']]
            inv_transform_matrices_tgt = compensate_pose(inv_transform_matrices, tgt_pose)

        stabilized_pairs = []
        corresponding_displ = []
        for i in range(seq_length):
            if i == sample['tgt_index']:
                continue
            img = imgs[:,i]
            img_pose = inv_transform_matrices_tgt[:,i]
            stab_img = inverse_rotate(img, img_pose[:,:,:3], intrinsics)
            pair = torch.cat([stab_img, tgt_img], dim=1)  # [1, 6, H, W]
            stabilized_pairs.append(pair)

            GT_translations = sample['poses'][:,:,-1]
            real_displacement = np.linalg.norm(GT_translations[sample['tgt_index']] - GT_translations[i])
            corresponding_displ.append(real_displacement)
        stab_batch = torch.cat(stabilized_pairs)  # [seq, 6, H, W]
        depth_maps = depth_net(stab_batch)  # [seq, 1 , H/4, W/4]

        selected_depth, selected_index = select_best_map(depth_maps, target_mean_depthnet_output)

        pred_depth = selected_depth * corresponding_displ[selected_index] / args.nominal_displacement

        if args.save_output:
            if j == 0:
                predictions = np.zeros((len(test_files), *pred_depth.shape))
            predictions[j] = 1/pred_depth

        gt_depth = sample['gt_depth']
        pred_depth_zoomed = F.interpolate(pred_depth.view(1,1,*pred_depth.shape),
                                          gt_depth.shape[:2],
                                          mode='bilinear',
                                          align_corners=False).clamp(args.min_depth, args.max_depth)[0,0]
        if sample['mask'] is not None:
            pred_depth_zoomed_masked = pred_depth_zoomed.cpu().numpy()[sample['mask']]
            gt_depth = gt_depth[sample['mask']]
        errors[:,j] = compute_errors(gt_depth, pred_depth_zoomed_masked)
        if args.log_best_worst:
            if best_error > errors[0,j]:
                best_error = errors[0,j]
                log_result(pred_depth_zoomed, sample['gt_depth'], stab_batch, selected_index, args.output_dir, 'best')
            if worst_error < errors[0,j]:
                worst_error = errors[0,j]
                log_result(pred_depth_zoomed, sample['gt_depth'], stab_batch, selected_index, args.output_dir, 'worst')

    mean_errors = errors.mean(1)
    error_names = ['mean_abs', 'abs_rel','abs_log','sq_rel','rms','log_rms','a1','a2','a3']

    print("Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors))

    if args.save_output:
        np.save(args.output_dir/'predictions.npy', predictions)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    mabs = np.mean(np.abs(gt - pred))
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_log = np.mean(np.abs(np.log(gt) - np.log(pred)))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return mabs, abs_rel, abs_log, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__ == '__main__':
    main()
