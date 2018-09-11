from __future__ import division

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, zeros_, constant_


def conv(in_planes, out_planes, kernel_size=3, stride=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True)
        )


def deconv(in_planes, out_planes, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )


def predict_depth(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


def post_process_depth(out, activation_function=None):
    depth = out
    if activation_function is not None:
        depth = activation_function(depth)

    return depth


def adaptative_cat(out_conv, out_deconv, out_depth_up):
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)


def init_modules(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            constant_(m.weight, 1)
            zeros_(m.bias)