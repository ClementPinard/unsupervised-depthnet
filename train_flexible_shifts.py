import time

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import models
import train_img_pairs
from inverse_warp import compensate_pose, pose_vec2mat, inverse_rotate
from logger import AverageMeter

train_img_pairs.parser.add_argument('-d', '--target-mean-depth', type=float,
                                    help='equivalent depth to aim at when adjustting shifts, regarding DepthNet output',
                                    metavar='D', default=40)
train_img_pairs.parser.add_argument('-r', '--recompute-frequency', type=int,
                                    help='Will recompute optimal shifts every R epochs',
                                    metavar='R', default=5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    env = train_img_pairs.prepare_environment()
    env['adjust_loader'] = torch.utils.data.DataLoader(
        env['train_set'], batch_size=env['args'].batch_size, shuffle=False,
        num_workers=0, pin_memory=True)  # workers is set to 0 to avoid multiple instances to be modified at the same time
    launch_training_flexible_shifts(**env)


def launch_training_flexible_shifts(scheduler, **env):
    logger = env['logger']
    args = env["args"]
    train_set = env["train_set"]
    env['best_error'] = -1
    env['epoch'] = 0
    env['n_iter'] = 0

    if args.pretrained_depth or args.evaluate:
        train_img_pairs.validate(**env)

    for epoch in range(1, args.epochs + 1):
        env['epoch'] = epoch
        scheduler.step()
        logger.epoch_bar.update(epoch)

        # train for one epoch
        train_loss, env['n_iter'] = train_img_pairs.train_one_epoch(**env)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        if epoch % args.recompute_frequency == 0:
            train_set.adjust = True
            average_shifts = adjust_shifts(**env)
            shifts_string = ' '.join(['{:.3f}'.format(s) for s in average_shifts])
            logger.train_writer.write(' * adjusted shifts, average shifts are now : {}'.format(shifts_string))
            train_set.adjust = False

        # evaluate on validation set
        error = train_img_pairs.validate(**env)

        env['best_error'] = train_img_pairs.finish_epoch(train_loss, error, **env)
    logger.epoch_bar.finish()


@torch.no_grad()
def adjust_shifts(args, train_set, adjust_loader, depth_net, pose_net, epoch, logger, training_writer, **env):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    new_shifts = AverageMeter(args.sequence_length-1, precision=2)
    pose_net.eval()
    depth_net.eval()
    upsample_depth_net = models.UpSampleNet(depth_net, args.network_input_size)

    end = time.time()

    mid_index = (args.sequence_length - 1)//2

    # we contrain mean value of depth net output from pair 0 and mid_index
    target_values = np.arange(-mid_index, mid_index + 1) / (args.target_mean_depth * mid_index)
    target_values = 1/np.abs(np.concatenate([target_values[:mid_index], target_values[mid_index + 1:]]))

    logger.reset_train_bar(len(adjust_loader))

    for i, sample in enumerate(adjust_loader):
        index = sample['index']

        # measure data loading time
        data_time.update(time.time() - end)
        imgs = torch.stack(sample['imgs'], dim=1).to(device)
        intrinsics = sample['intrinsics'].to(device)
        intrinsics_inv = sample['intrinsics_inv'].to(device)

        # compute output
        batch_size, seq = imgs.size()[:2]

        if args.network_input_size is not None:
            h,w = args.network_input_size
            downsample_imgs = F.interpolate(imgs,
                                            (3, h, w),
                                            mode='area')
            poses = pose_net(downsample_imgs)  # [B, seq, 6]
        else:
            poses = pose_net(imgs)

        pose_matrices = pose_vec2mat(poses, args.rotation_mode)  # [B, seq, 3, 4]

        tgt_imgs = imgs[:, mid_index]  # [B, 3, H, W]
        tgt_poses = pose_matrices[:, mid_index]  # [B, 3, 4]
        compensated_poses = compensate_pose(pose_matrices, tgt_poses)  # [B, seq, 3, 4] tgt_poses are now neutral pose

        ref_indices = list(range(args.sequence_length))
        ref_indices.remove(mid_index)

        mean_depth_batch = []

        for ref_index in ref_indices:
            prior_imgs = imgs[:, ref_index]
            prior_poses = compensated_poses[:, ref_index]  # [B, 3, 4]

            prior_imgs_compensated = inverse_rotate(prior_imgs, prior_poses[:,:,:3], intrinsics, intrinsics_inv)
            input_pair = torch.cat([prior_imgs_compensated, tgt_imgs], dim=1)  # [B, 6, W, H]

            depth = upsample_depth_net(input_pair)  # [B, 1, H, W]
            mean_depth = depth.view(batch_size, -1).mean(-1).cpu().numpy()  # B
            mean_depth_batch.append(mean_depth)

        for j, mean_values in zip(index, np.stack(mean_depth_batch, axis=-1)):
            ratio = mean_values / target_values  # if mean value is too high, raise the shift, lower otherwise
            train_set.reset_shifts(j, ratio[:mid_index], ratio[mid_index:])
            new_shifts.update(train_set.get_shifts(j))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.train_bar.update(i)
        if i % args.print_freq == 0:
            logger.train_writer.write('Adjustement:'
                                      'Time {} Data {} shifts {}'.format(batch_time, data_time, new_shifts))

    for i, shift in enumerate(new_shifts.avg):
            training_writer.add_scalar('shifts{}'.format(i), shift, epoch)

    return new_shifts.avg


if __name__ == '__main__':
    main()
