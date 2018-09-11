import torch
import models
import argparse


parser = argparse.ArgumentParser(description='DepthNet BN to DepthNet converter',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('depth_bn_path', metavar='PATH',
                    help='path to depthnet bn weights')
parser.add_argument('--depth_path', default='depthnet.pth.tar', metavar='PATH',
                    help='where to save depthnet weights')
args = parser.parse_args()
eps = 1e-3
depthnet = models.DepthNet().cuda()

depth_bn = torch.load(args.depth_bn_path)

depth_bn_state = depth_bn['state_dict']

depthnet.load_state_dict(depth_bn_state, strict=False)

state_dict = depthnet.state_dict()

for k in depth_bn_state.keys():
    if 'running_mean' in k:
        layer, index, _ = k.split('.')
        rm = depth_bn_state['.'.join([layer, index, 'running_mean'])]
        rv = depth_bn_state['.'.join([layer, index, 'running_var'])]
        w = depth_bn_state['.'.join([layer, index, 'weight'])]
        b = depth_bn_state['.'.join([layer, index, 'bias'])]

        conv_w = state_dict['.'.join([layer, str(int(index)-1), 'weight'])]
        conv_b = state_dict['.'.join([layer, str(int(index)-1), 'bias'])]

        inv_std = (rv + eps).pow(-0.5)

        conv_w.mul_(inv_std.view(conv_w.size(0), 1, 1, 1))
        conv_b.add_(-rm).mul_(inv_std)
        conv_w.mul_(w.view(conv_w.size(0), 1, 1, 1))
        conv_b.mul_(w).add_(b)

depth_bn['state_dict'] = state_dict
depth_bn['bn'] = False
torch.save(depth_bn, args.depth_path)