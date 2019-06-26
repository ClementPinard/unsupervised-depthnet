import argparse
import time
import csv

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import custom_transforms
import models

from collections import OrderedDict

from utils import tensor2array, save_checkpoint, save_path_formatter, log_output_tensorboard
from inverse_warp import compensate_pose, pose_vec2mat, inverse_rotate, invert_mat

from loss_functions import photometric_reconstruction_loss, compute_depth_errors, compute_pose_error, grad_diffusion_loss
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-format', default='KITTI', choices=['KITTI', 'StillBox'])
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--nominal-displacement', type=float, metavar='D', default=0.3,
                    help='magnitude assumption of DepthNet when given a pair of frames')
parser.add_argument('--supervise-pose', action='store_true',
                    help='use avalaible gt pose to supervise posenet and perform rotation compensation')
parser.add_argument('--network-input-size', type=int, nargs=2, default=None,
                    help='size to which images have to be resized before def into network, can only be smaller than raw image size. \
                    if not set, will take raw image size')
parser.add_argument('--upscale', action='store_true', help='upscale depth maps from network to match image size \
                    if not set, will downscale images to match depth maps')
parser.add_argument('--same-ratio', default=0, type=float, metavar='P', help='probability to pick pairs with the same image, compared to others\
                    Only effective after first milestone')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store depth in npy 2D arrays and pose in 12 columns csv. See data/kitti_raw_loader.py for an example')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--training-milestones', default=[10,20], type=int, metavar='N', nargs=2,
                    help='epochs at which training switch modes')
parser.add_argument('--lr-decay-frequency', '--lr-df', default=50, type=int, metavar='N',
                    help='will divide lr by 2 every N epoch')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bn', choices=['none','pose','depth','both'], default='none',
                    metavar='W', help='To which network batch norm is applied')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained-depth', dest='pretrained_depth', default=None, metavar='PATH',
                    help='path to pre-trained DepthNet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH',
                    help='path to pre-trained Pose net model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('--ssim', type=float, help='weight for SSIM loss', metavar='W', default=0.1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=30)
parser.add_argument('--kappa', default=1, type=float, help='kappa parameter for diffusion')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--max-depth', type=float, help='value to which depth colormap will be capped to', metavar='M', default=100)
parser.add_argument('-f', '--training-output-freq', type=int, metavar='N', default=0,
                    help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    env = prepare_environment()
    launch_training(**env)


def prepare_environment():
    env = {}
    args = parser.parse_args()
    if args.dataset_format == 'KITTI':
        from datasets.shifted_sequence_folders import ShiftedSequenceFolder
    elif args.dataset_format == 'StillBox':
        from datasets.shifted_sequence_folders import StillBox as ShiftedSequenceFolder
    elif args.dataset_format == 'TUM':
        from datasets.shifted_sequence_folders import TUM as ShiftedSequenceFolder
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    args.test_batch_size = 4*args.batch_size
    if args.evaluate:
        args.epochs = 0

    env['tb_writer'] = SummaryWriter(args.save_path)
    env['sample_nb_to_log'] = 3

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        # custom_transforms.RandomHorizontalFlip(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = ShiftedSequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        with_depth_gt=False,
        with_pose_gt=args.supervise_pose,
        sequence_length=args.sequence_length
    )
    val_set = ShiftedSequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        with_depth_gt=args.with_gt,
        with_pose_gt=args.with_gt
    )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=4*args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    env['train_set'] = train_set
    env['val_set'] = val_set
    env['train_loader'] = train_loader
    env['val_loader'] = val_loader

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    pose_net = models.PoseNet(seq_length=args.sequence_length,
                              batch_norm=args.bn in ['pose', 'both'],
                              input_size=args.network_input_size).to(device)

    if args.pretrained_pose:
        print("=> using pre-trained weights for pose net")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    depth_net = models.DepthNet(depth_activation="elu",
                                batch_norm=args.bn in ['depth', 'both'],
                                input_size=args.network_input_size,
                                upscale=args.upscale).to(device)

    if args.pretrained_depth:
        print("=> using pre-trained DepthNet model")
        data = torch.load(args.pretrained_depth)
        depth_net.load_state_dict(data['state_dict'])

    cudnn.benchmark = True
    depth_net = torch.nn.DataParallel(depth_net)
    pose_net = torch.nn.DataParallel(pose_net)

    env['depth_net'] = depth_net
    env['pose_net'] = pose_net

    print('=> setting adam solver')

    optim_params = [
        {'params': depth_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    # parameters = chain(depth_net.parameters(), pose_exp_net.parameters())
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                args.lr_decay_frequency,
                                                gamma=0.5)
    env['optimizer'] = optimizer
    env['scheduler'] = scheduler

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()
    env['logger'] = logger

    env['args'] = args

    return env


def launch_training(scheduler, **env):
    logger = env['logger']
    args = env["args"]
    env['best_error'] = -1
    env['epoch'] = 0
    env['n_iter'] = 0

    if args.pretrained_depth or args.evaluate:
        validate(**env)

    for epoch in range(1, args.epochs + 1):
        env['epoch'] = epoch
        scheduler.step()
        logger.epoch_bar.update(epoch)

        # train for one epoch
        train_loss, env['n_iter'] = train_one_epoch(**env)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        error = validate(**env)

        env['best_error'] = finish_epoch(train_loss, error, **env)
    logger.epoch_bar.finish()


def finish_epoch(train_loss, error, best_error, args, epoch, depth_net, pose_net, **env):
    if best_error < 0:
        best_error = error

    # remember lowest error and save checkpoint
    is_best = error < best_error
    best_error = min(best_error, error)
    save_checkpoint(
        args.save_path, {
            'epoch': epoch,
            'state_dict': depth_net.module.state_dict(),
            'bn': args.bn in ['depth', 'both'],
            'nominal_displacement': args.nominal_displacement
        }, {
            'epoch': epoch,
            'bn': args.bn in ['pose', 'both'],
            'state_dict': pose_net.module.state_dict()
        },
        is_best)

    with open(args.save_path/args.log_summary, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([train_loss, error])
    return best_error


def train_one_epoch(args, train_loader,
                    depth_net, pose_net, optimizer,
                    epoch, n_iter,
                    logger, tb_writer, **env):
    global device
    logger.reset_train_bar()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.ssim
    e1, e2 = args.training_milestones

    # switch to train mode
    depth_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, sample in enumerate(train_loader):

        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        imgs = torch.stack(sample['imgs'], dim=1).to(device)
        intrinsics = sample['intrinsics'].to(device)

        batch_size, seq = imgs.size()[:2]

        if args.network_input_size is not None:
            h,w = args.network_input_size
            downsample_imgs = F.interpolate(imgs,(3, h, w), mode='area')
            poses = pose_net(downsample_imgs)  # [B, seq, 6]
        else:
            poses = pose_net(imgs)

        pose_matrices = pose_vec2mat(poses, args.rotation_mode)  # [B, seq, 3, 4]

        total_indices = torch.arange(seq, dtype=torch.int64, device=device).expand(batch_size, seq)
        batch_range = torch.arange(batch_size, dtype=torch.int64, device=device)

        ''' for each element of the batch select a random picture in the sequence to
        which we will compute the depth, all poses are then converted so that pose of this
        very picture is exactly identity. At first this image is always in the middle of the sequence'''

        if epoch > e2:
            tgt_id = torch.randint(0, seq, (batch_size,), device=device)
        else:
            tgt_id = torch.full_like(batch_range, args.sequence_length//2)

        ref_ids = total_indices[total_indices != tgt_id.unsqueeze(1)].view(batch_size, seq - 1)

        '''
        Select what other picture we are going to feed DepthNet, it must not be the same
        as tgt_id. At first, it's always first picture of the sequence, it is randomly chosen when first training milestone is reached
        '''

        if epoch > e1:
            probs = torch.ones_like(total_indices, dtype=torch.float32)
            probs[batch_range, tgt_id] = args.same_ratio
            prior_id = torch.multinomial(probs, 1)[:,0]
        else:
            prior_id = torch.zeros_like(batch_range)

        # Treat the case of prior_id == tgt_id and the depth must be max_depth, regardless of apparent movement

        tgt_imgs = imgs[batch_range, tgt_id]  # [B, 3, H, W]
        tgt_poses = pose_matrices[batch_range, tgt_id]  # [B, 3, 4]

        prior_imgs = imgs[batch_range, prior_id]

        compensated_poses = compensate_pose(pose_matrices, tgt_poses)  # [B, seq, 3, 4] tgt_poses are now neutral pose
        prior_poses = compensated_poses[batch_range, prior_id]  # [B, 3, 4]

        if args.supervise_pose:
            from_GT = invert_mat(sample['pose']).to(device)
            compensated_GT_poses = compensate_pose(from_GT, from_GT[batch_range, tgt_id])
            prior_GT_poses = compensated_GT_poses[batch_range, prior_id]
            prior_imgs_compensated = inverse_rotate(prior_imgs, prior_GT_poses[:,:,:-1], intrinsics)
        else:
            prior_imgs_compensated = inverse_rotate(prior_imgs, prior_poses[:,:,:-1], intrinsics)

        input_pair = torch.cat([prior_imgs_compensated, tgt_imgs], dim=1)  # [B, 6, W, H]
        depth = depth_net(input_pair)

        # depth = [sample['depth'].to(device).unsqueeze(1) * 3 / abs(tgt_id[0] - prior_id[0])]
        # depth.append(torch.nn.functional.interpolate(depth[0], scale_factor=2))
        disparities = [1/d for d in depth]

        predicted_magnitude = prior_poses[:, :, -1:].norm(p=2, dim=1, keepdim=True).unsqueeze(1)
        scale_factor = args.nominal_displacement / (predicted_magnitude + 1e-5)
        normalized_translation = compensated_poses[:, :, :, -1:] * scale_factor  # [B, seq_length-1, 3]
        new_pose_matrices = torch.cat([compensated_poses[:, :, :, :-1], normalized_translation], dim=-1)

        biggest_scale = depth[0].size(-1)

        # Construct valid sequence to compute photometric error,
        # make the rest converge to max_depth because nothing moved
        vb = batch_range[prior_id != tgt_id]
        same_range = batch_range[prior_id == tgt_id]  # batch of still pairs

        loss_1 = 0
        loss_1_same = 0
        for k, scaled_depth in enumerate(depth):
            size_ratio = scaled_depth.size(-1) / biggest_scale

            if len(same_range) > 0:
                still_depth = scaled_depth[same_range]
                loss_same = F.smooth_l1_loss(still_depth/args.max_depth, torch.ones_like(still_depth))
            else:
                loss_same = 0

            loss_valid, *to_log = photometric_reconstruction_loss(imgs[vb], tgt_id[vb], ref_ids[vb],
                                                                  scaled_depth[vb], new_pose_matrices[vb],
                                                                  intrinsics[vb],
                                                                  args.rotation_mode,
                                                                  ssim_weight=w3,
                                                                  upsample=args.upscale)

            loss_1 += loss_valid * size_ratio
            loss_1_same += loss_same * size_ratio

            if log_output and len(vb) > 0:
                log_output_tensorboard(tb_writer, "train", 0, k, n_iter,
                                       scaled_depth[0], disparities[k][0],
                                       *to_log)
        loss_2 = grad_diffusion_loss(disparities, tgt_imgs, args.kappa)

        loss = w1*(loss_1 + loss_1_same) + w2*loss_2
        if args.supervise_pose:
            loss += (from_GT[:,:,:,:3] - pose_matrices[:,:,:,:3]).abs().mean()

        if log_losses:
            tb_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            tb_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            tb_writer.add_scalar('total_loss', loss.item(), n_iter)

        if log_output and len(vb) > 0:
            valid_poses = poses[vb]
            nominal_translation_magnitude = valid_poses[:,-2,:3].norm(p=2, dim=-1)
            # Log the translation magnitude relative to translation magnitude between last and penultimate frames
            # for a perfectly constant displacement magnitude, you should get ratio of 2,3,4 and so forth.
            # last pose is always identity and penultimate translation magnitude is always 1, so you don't need to log them
            for j in range(args.sequence_length - 2):
                trans_mag = valid_poses[:,j,:3].norm(p=2, dim=-1)
                tb_writer.add_histogram('tr {}'.format(j),
                                        (trans_mag/nominal_translation_magnitude).detach().cpu().numpy(),
                                        n_iter)
            for j in range(args.sequence_length - 1):
                # TODO log a better value : this is magnitude of vector (yaw, pitch, roll) which is not a physical value
                rot_mag = valid_poses[:,j,3:].norm(p=2, dim=-1)
                tb_writer.add_histogram('rot {}'.format(j),
                                        rot_mag.detach().cpu().numpy(),
                                        n_iter)

            tb_writer.add_image('train Input', tensor2array(tgt_imgs[0]), n_iter)

        # record loss for average meter
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= args.epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0], n_iter


@torch.no_grad()
def validate(tb_writer, **env):
    env['logger'].reset_valid_bar()
    if env['args'].with_gt:
        errors = validate_with_gt(tb_writer=tb_writer, **env)
        errors_to_log = list(errors.items())[2:9]
        decisive_error = errors["stab abs log"]
    else:
        errors = validate_without_gt(**env)
        errors_to_log = errors.items()
        decisive_error = errors["Total Loss"]

    for name, error in errors.items():
        tb_writer.add_scalar(name, error, env['epoch'])

    error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in errors_to_log)
    env['logger'].valid_writer.write(' * Avg {}'.format(error_string))

    return decisive_error


def validate_without_gt(args, val_loader, depth_net, pose_net, epoch, logger, tb_writer, sample_nb_to_log, **env):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.ssim
    if args.log_output:
        poses_values = np.zeros(((len(val_loader) - 1) * args.test_batch_size * (args.sequence_length-1),6))
        disp_values = np.zeros(((len(val_loader) - 1) * args.test_batch_size * 3))

    # switch to evaluate mode
    depth_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

    for i, sample in enumerate(val_loader):
        log_output = i < sample_nb_to_log

        imgs = torch.stack(sample['imgs'], dim=1).to(device)
        intrinsics = sample['intrinsics'].to(device)

        if epoch == 1 and log_output:
            for j,img in enumerate(sample['imgs']):
                tb_writer.add_image('val Input/{}'.format(i), tensor2array(img[0]), j)

        batch_size, seq = imgs.size()[:2]
        poses = pose_net(imgs)
        pose_matrices = pose_vec2mat(poses, args.rotation_mode)  # [B, seq, 3, 4]

        mid_index = (args.sequence_length - 1)//2

        tgt_imgs = imgs[:, mid_index]  # [B, 3, H, W]
        tgt_poses = pose_matrices[:, mid_index]  # [B, 3, 4]
        compensated_poses = compensate_pose(pose_matrices, tgt_poses)  # [B, seq, 3, 4] tgt_poses are now neutral pose

        ref_ids = list(range(args.sequence_length))
        ref_ids.remove(mid_index)

        loss_1 = 0
        loss_2 = 0

        for ref_index in ref_ids:
            prior_imgs = imgs[:, ref_index]
            prior_poses = compensated_poses[:, ref_index]  # [B, 3, 4]

            prior_imgs_compensated = inverse_rotate(prior_imgs, prior_poses[:,:,:3], intrinsics)
            input_pair = torch.cat([prior_imgs_compensated, tgt_imgs], dim=1)  # [B, 6, W, H]

            predicted_magnitude = prior_poses[:, :, -1:].norm(p=2, dim=1, keepdim=True).unsqueeze(1)  # [B, 1, 1, 1]
            scale_factor = args.nominal_displacement / predicted_magnitude
            normalized_translation = compensated_poses[:, :, :, -1:] * scale_factor  # [B, seq, 3, 1]
            new_pose_matrices = torch.cat([compensated_poses[:, :, :, :-1], normalized_translation], dim=-1)

            depth = depth_net(input_pair)
            disparity = 1/depth

            tgt_id = torch.full((batch_size,), ref_index, dtype=torch.int64, device=device)
            ref_ids_tensor = torch.tensor(ref_ids, dtype=torch.int64, device=device).expand(batch_size, -1)
            photo_loss, *to_log = photometric_reconstruction_loss(imgs, tgt_id, ref_ids_tensor,
                                                                  depth, new_pose_matrices,
                                                                  intrinsics,
                                                                  args.rotation_mode,
                                                                  ssim_weight=w3, upsample=args.upscale)

            loss_1 += photo_loss

            if log_output:
                log_output_tensorboard(tb_writer, "train", i, ref_index, epoch,
                                       depth[0], disparity[0],
                                       *to_log)

            loss_2 += grad_diffusion_loss(disparity, tgt_imgs, args.kappa)

        if args.log_output and i < len(val_loader)-1:
            step = args.test_batch_size * (args.sequence_length-1)
            poses_values[i * step:(i+1) * step] = poses[:, :-1].cpu().view(-1,6).numpy()
            step = args.test_batch_size * 3
            disp_unraveled = disparity.cpu().view(args.test_batch_size, -1)
            disp_values[i * step:(i+1) * step] = torch.cat([disp_unraveled.min(-1)[0],
                                                            disp_unraveled.median(-1)[0],
                                                            disp_unraveled.max(-1)[0]]).numpy()

        loss = w1*loss_1 + w2*loss_2
        losses.update([loss.item(), loss_1.item(), loss_2.item()])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    if args.log_output:
        rot_coeffs = ['rx', 'ry', 'rz'] if args.rotation_mode == 'euler' else ['qx', 'qy', 'qz']
        tr_coeffs = ['tx', 'ty', 'tz']
        for k, (coeff_name) in enumerate(tr_coeffs + rot_coeffs):
            tb_writer.add_histogram('val poses_{}'.format(coeff_name), poses_values[:,k], epoch)
        tb_writer.add_histogram('disp_values', disp_values, epoch)
    logger.valid_bar.update(len(val_loader))
    return OrderedDict(zip(['Total loss', 'Photo loss', 'Smooth loss'], losses.avg))


def validate_with_gt(args, val_loader, depth_net, pose_net, epoch, logger, tb_writer, sample_nb_to_log, **env):
    global device
    batch_time = AverageMeter()
    depth_error_names = ['abs diff', 'abs rel', 'abs log', 'a1', 'a2', 'a3']
    stab_depth_errors = AverageMeter(i=len(depth_error_names))
    unstab_depth_errors = AverageMeter(i=len(depth_error_names))
    pose_error_names = ['Absolute Trajectory Error', 'Rotation Error']
    pose_errors = AverageMeter(i=len(pose_error_names))

    # switch to evaluate mode
    depth_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, sample in enumerate(val_loader):
        log_output = i < sample_nb_to_log

        imgs = torch.stack(sample['imgs'], dim=1).to(device)

        intrinsics = sample['intrinsics'].to(device)

        GT_depth = sample['depth'].to(device)
        GT_pose = sample['pose'].to(device)

        batch_size, seq, c, h, w = imgs.shape
        dh, dw = GT_depth.shape[-2:]

        mid_index = (args.sequence_length - 1)//2

        tgt_img = imgs[:,mid_index]

        if epoch == 1 and log_output:
                for j,img in enumerate(sample['imgs']):
                    tb_writer.add_image('val Input/{}'.format(i), tensor2array(img[0]), j)
                depth_to_show = GT_depth[0].cpu()
                # KITTI Like data routine to discard invalid data
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0,10)
                tb_writer.add_image('val target Disparity Normalized/{}'.format(i),
                                    tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                    epoch)

        poses = pose_net(imgs)
        pose_matrices = pose_vec2mat(poses, args.rotation_mode)  # [B, seq, 3, 4]
        inverted_pose_matrices = invert_mat(pose_matrices)
        ATE, RE = compute_pose_error(GT_pose[:,:-1], inverted_pose_matrices[:,:-1])
        pose_errors.update([ATE.item(), RE.item()])

        tgt_poses = pose_matrices[:, mid_index]  # [B, 3, 4]
        compensated_predicted_poses = compensate_pose(pose_matrices, tgt_poses)
        compensated_GT_poses = compensate_pose(GT_pose, GT_pose[:,mid_index])

        for j in range(args.sequence_length):
            if j == mid_index:
                if log_output and epoch == 1:
                    tb_writer.add_image('val Input Stabilized/{}'.format(i), tensor2array(sample['imgs'][j][0]), j)
                continue

            '''compute displacement magnitude for each element of batch, and rescale
            depth accordingly.'''

            prior_img = imgs[:,j]
            displacement = compensated_GT_poses[:, j, :, -1]  # [B,3]
            displacement_magnitude = displacement.norm(p=2, dim=1)  # [B]
            current_GT_depth = (GT_depth * args.nominal_displacement / displacement_magnitude.view(-1, 1, 1)).clamp(0,args.max_depth)

            prior_predicted_pose = compensated_predicted_poses[:, j]  # [B, 3, 4]
            prior_GT_pose = compensated_GT_poses[:, j]

            prior_predicted_rot = prior_predicted_pose[:,:,:-1]
            prior_GT_rot = prior_GT_pose[:,:,:-1].transpose(1,2)

            prior_compensated_from_GT = inverse_rotate(prior_img,
                                                       prior_GT_rot,
                                                       intrinsics)
            if log_output and epoch == 1:
                depth_to_show = current_GT_depth[0]
                tb_writer.add_image('val target Depth {}/{}'.format(j, i), tensor2array(depth_to_show, max_value=args.max_depth), epoch)
                tb_writer.add_image('val Input Stabilized/{}'.format(i), tensor2array(prior_compensated_from_GT[0]), j)

            prior_compensated_from_prediction = inverse_rotate(prior_img, prior_predicted_rot, intrinsics)
            predicted_input_pair = torch.cat([prior_compensated_from_prediction, tgt_img], dim=1)  # [B, 6, W, H]
            GT_input_pair = torch.cat([prior_compensated_from_GT, tgt_img], dim=1)  # [B, 6, W, H]

            # This is the depth from footage stabilized with GT pose, it should be better than depth from raw footage without any GT info
            raw_depth_stab = depth_net(GT_input_pair)
            raw_depth_unstab = depth_net(predicted_input_pair)

            # Upsample depth so that it matches GT size
            depth_stab = F.interpolate(raw_depth_stab, (dh, dw), mode='bilinear', align_corners=False)
            depth_unstab = F.interpolate(raw_depth_unstab, (dh, dw), mode='bilinear', align_corners=False)

            for k, depth in enumerate([depth_stab, depth_unstab]):
                disparity = 1/depth
                errors = stab_depth_errors if k == 0 else unstab_depth_errors
                errors.update(compute_depth_errors(current_GT_depth, depth, crop=True, max_depth=101))
                if log_output:
                    prefix = 'stabilized' if k == 0 else 'unstabilized'
                    tb_writer.add_image('val {} Dispnet Output Normalized {}/{}'.format(prefix, j, i),
                                        tensor2array(disparity[0],max_value=None, colormap='magma'),
                                        epoch)
                    tb_writer.add_image('val {} Depth Output {}/{}'.format(prefix, j, i),
                                        tensor2array(depth[0], max_value=args.max_depth),
                                        epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(
                'valid: Time {} ATE Error {:.4f} ({:.4f}), Unstab Rel Abs Error {:.4f} ({:.4f})'.format(
                    batch_time, pose_errors.val[0], pose_errors.avg[0],
                    unstab_depth_errors.val[1], unstab_depth_errors.avg[1])
            )
    logger.valid_bar.update(len(val_loader))

    errors = (*pose_errors.avg,
              *unstab_depth_errors.avg,
              *stab_depth_errors.avg)
    error_names = (*pose_error_names,
                   *['unstab {}'.format(e) for e in depth_error_names],
                   *['stab {}'.format(e) for e in depth_error_names])

    return OrderedDict(zip(error_names, errors))


if __name__ == '__main__':
    main()
