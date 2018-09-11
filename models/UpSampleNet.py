from __future__ import division
import torch.nn as nn
import torch.nn.functional as F


class UpSampleNet(nn.Module):

    def __init__(self, network, input_size=None):
        super(UpSampleNet, self).__init__()
        self.network = network
        self.input_size = input_size

    def forward(self, x):
        x_size = x.size()[-2:]
        if self.input_size is None:
            self.input_size = x_size
        downscaled_x = F.interpolate(x, self.input_size, mode='area')
        output = self.network(downscaled_x)

        if isinstance(output, tuple):
            return (F.interpolate(output[0], x_size, mode='bilinear', align_corners=False), *output)

        else:
            return F.interpolate(output, x_size, mode='bilinear', align_corners=False)