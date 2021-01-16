from path import Path
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import torch
from torch.nn import functional as F
from models import DepthNet, PoseNet
from skimage.transform import resize
from evaluation_toolkit.inference_toolkit import inferenceFramework
from inverse_warp import inverse_rotate


@torch.no_grad()
def main():
    parser = ArgumentParser(description='Example usage of Inference toolkit',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_root', metavar='DIR', type=Path, required=True)
    parser.add_argument('--depth_output', metavar='FILE', type=Path, required=True,
                        help='where to store the estimated depth maps, must be a npy file')
    parser.add_argument('--evaluation_list_path', metavar='PATH', type=Path, required=True,
                        help='File with list of images to test for depth evaluation')
    parser.add_argument('--pretrained_depthnet', metavar='FILE', type=Path, required=True)
    parser.add_argument('--no-resize', action='store_true')
    parser.add_argument("--img-height", default=128, type=int, help="Image height")
    parser.add_argument("--img-width", default=416, type=int, help="Image width")
    parser.add_argument("--nominal-displacement", "-D", type=float, default=0.3)
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open(args.evaluation_list_path) as f:
        evaluation_list = [line[:-1] for line in f.readlines()]

    def preprocessing(frame):
        h, w, _ = frame.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            frame = resize(frame, (args.img_height, args.img_width))
        frame_np = (frame.transpose(2, 0, 1).astype(np.float32)[None]/255 - 0.5)/0.5
        return torch.from_numpy(frame_np).to(device)

    engine = inferenceFramework(args.dataset_root, evaluation_list, frame_transform=preprocessing, max_shift=10)

    depth_net = DepthNet().to(device)
    weights = torch.load(args.pretrained_depthnet)
    depth_net.load_state_dict(weights['state_dict'])
    depth_net.eval()

    if args.pretrained_posenet is not None:
        weights = torch.load(args.pretrained_posenet)
        seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
        pose_net = PoseNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    for sample in tqdm(engine):
        tgt_img, latest_intrinsics, poses = sample.get_frame()
        ref_img, _, previous_pose = sample.get_previous_frame(displacement=args.nominal_displacement)

        previous_pose = torch.from_numpy(previous_pose).to(ref_img)
        inv_rot = previous_pose[:, :3].T
        latest_intrinsics = torch.from_numpy(latest_intrinsics).to(ref_img)

        stab_img = inverse_rotate(ref_img, inv_rot[None], latest_intrinsics)
        pair = torch.cat([stab_img, tgt_img], dim=1)  # [1, 6, H, W]

        pred_depth = depth_net(pair)

        scale_factor = np.linalg.norm(previous_pose[:, 3]) / args.nominal_displacement

        pred_depth *= scale_factor

        if (not args.no_resize) and (pred_depth.shape[0] != args.img_height or pred_depth.shape[1] != args.img_width):
            out_shape = (args.img_height, args.img_width)
        else:
            out_shape = tgt_img.shape[:2]

        pred_depth_zoomed = F.interpolate(pred_depth.view(1, 1, *pred_depth.shape),
                                          out_shape,
                                          mode='bilinear',
                                          align_corners=False)
        pred_depth_zoomed = pred_depth_zoomed.cpu().numpy()[0, 0]

        engine.finish_frame(pred_depth_zoomed)
    mean_inference_time, output_depth_maps = engine.finalize(output_path=args.depth_output)

    print("Mean time per sample : {:.2f}us".format(1e6 * mean_inference_time))
    np.savez(args.depth_output, **output_depth_maps)


if __name__ == '__main__':
    main()
