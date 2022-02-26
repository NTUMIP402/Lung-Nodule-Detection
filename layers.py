#!/usr/bin/python3
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Simple residual block
    Design inspiraed by Kaiming's 'Identity Mappings in Deep Residual Networks'
    https://arxiv.org/pdf/1603.05027v3.pdf
    """

    def __init__(self, last_planes, in_planes, out_planes, kernel_size, stride, padding, dilation, debug=False):
        super(ResidualBlock, self).__init__()
        self.debug = debug
        self.last_planes = last_planes
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.n0 = nn.InstanceNorm3d(last_planes)
        self.conv1 = nn.Conv3d(last_planes, in_planes, kernel_size=1, bias=False)
        self.n1 = nn.InstanceNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.n2 = nn.InstanceNorm3d(in_planes)
        self.conv3 = nn.Conv3d(in_planes, out_planes, kernel_size=1, bias=False)
        self.shortcut = nn.Conv3d(last_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        if self.debug: print(f'bottleneck: x={x.size()}')
        if self.debug: print(f'last_planes={self.last_planes} '
                             f'in_planes={self.in_planes} '
                             f'out_planes={self.out_planes} '
                             )

        out = F.relu(self.n0(x))
        if self.debug: print(f'ResidualBlock:x={out.size()}')

        out = F.relu(self.n1(self.conv1(out)))
        if self.debug: print(f'ResidualBlock: conv1={out.size()}')

        out = F.relu(self.n2(self.conv2(out)))
        if self.debug: print(f'ResidualBlock: conv2={out.size()}')

        out = self.conv3(out)
        if self.debug: print(f'ResidualBlock: conv3={out.size()}')

        x = self.shortcut(x)
        if self.debug: print(f'ResidualBlock: shortcut={x.size()}')

        out = out + x
        if self.debug: print(f'ResidualBlock: conv3+shortcut={out.size()}')

        return out


class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output, thresh=-3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh
        xx, yy, zz, aa = np.where(mask)

        output = output[xx, yy, zz, aa]
        if ismask:
            return output, [xx, yy, zz, aa]
        else:
            return output
        # output = output[output[:, 0] >= self.conf_th]
        # bboxes = nms(output, self.nms_th)


class CReLU(nn.Module):
    """
    Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units
    arXiv:1603.05201
    https://arxiv.org/pdf/1603.05201.pdf
    """
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)
