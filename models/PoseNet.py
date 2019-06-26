import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import conv,init_modules


class PoseNet(nn.Module):

    def __init__(self, seq_length=3, batch_norm=False, input_size=None):
        super(PoseNet, self).__init__()
        self.seq_length = seq_length
        self.input_size = input_size

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3*self.seq_length, conv_planes[0], kernel_size=7, batch_norm=batch_norm, stride=2)
        self.conv2 = conv(conv_planes[0],    conv_planes[1], kernel_size=5, batch_norm=batch_norm, stride=2)
        self.conv3 = conv(conv_planes[1],    conv_planes[2],                batch_norm=batch_norm, stride=2)
        self.conv4 = conv(conv_planes[2],    conv_planes[3],                batch_norm=batch_norm, stride=2)
        self.conv5 = conv(conv_planes[3],    conv_planes[4],                batch_norm=batch_norm, stride=2)
        self.conv6 = conv(conv_planes[4],    conv_planes[5],                batch_norm=batch_norm, stride=2)
        self.conv7 = conv(conv_planes[5],    conv_planes[6],                batch_norm=batch_norm, stride=2)

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*(self.seq_length - 1), kernel_size=1, padding=0)
        init_modules(self)

    def forward(self, img_sequence):
        b, s, c, h, w = img_sequence.size()
        concatenated_imgs = img_sequence.view(b, s*c, h, w)

        if self.input_size:
            h,w = self.input_size
            concatenated_imgs = F.interpolate(concatenated_imgs,(h, w), mode='area')

        out_conv1 = self.conv1(concatenated_imgs)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = pose.view(pose.size(0), self.seq_length - 1, 6)
        pose = 0.1 * torch.cat([pose, pose[:,:1].detach()*0], dim=1)  # last frame is the Neutral position

        return pose