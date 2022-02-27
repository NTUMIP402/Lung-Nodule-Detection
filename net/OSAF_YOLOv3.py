import torch
import numpy as np
import math
import torch.nn as nn
from collections import OrderedDict
# from layers_se import *
from loss import Loss_recon, FocalLoss
import torch.nn.functional as F
from layers import GetPBB

config = {}
config['anchors'] = [5.0, 10.0, 20.]
config['channel'] = 1
config['crop_size'] = [80, 80, 80]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1.0
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 3. #mm, smallest nodule size
config['sizelim2'] = 10
config['sizelim3'] = 20
config['sizelim4'] = 30
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 0
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False, 'noise':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','990fbe3f0a1b53878669967b9afd1441','adc3bbc63d40f8761c59be10f1e504c3']
config['conf_thresh'] = 0.15


class Conv3d_WS(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x*(torch.tanh(F.softplus(x)))

class SpatialPyramidPooling(nn.Module):
    def __init__(self, feature_channels, pool_sizes=[3, 5]):
        super(SpatialPyramidPooling, self).__init__()

        # head conv
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
            Conv(feature_channels[-1] // 2, feature_channels[-1], 3),
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
        )

        self.maxpools = nn.ModuleList(
            [
                nn.MaxPool2d(pool_size, 1, pool_size // 2)
                for pool_size in pool_sizes
            ]
        )
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x] + features, dim=1)

        return features

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, pool_types=['avg', 'max', 'lse']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Conv3d(gate_channels, gate_channels, 1, 1, 0, bias=False),
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( max_pool )
            
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_3d(x).unsqueeze(-1).unsqueeze(-1)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).expand_as(x)
        return x * scale

def logsumexp_3d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, 3, 1, 1, bias=False),
            nn.BatchNorm3d(1)
        )
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class eSE(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch),
            nn.ReLU(inplace=True),
            nn.Linear(ch, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class conv3x3(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, act='mish', dilation=1):
        super().__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, stride, dilation=dilation, padding=dilation, bias=False)
        self.norm = nn.BatchNorm3d(out_ch)
        if act == 'mish':
            self.act = Mish()
        elif act == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input):
        return self.act(self.norm(self.conv(input)))

class conv1x1(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, act='mish'):
        super().__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 1, stride, padding=0, bias=False)
        self.norm = nn.BatchNorm3d(out_ch)
        if act == 'mish':
            self.act = Mish()
        elif act == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input):
        return self.act(self.norm(self.conv(input)))

class OSA_Module(nn.Module):

    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block, identity=False, SE=False):
        super().__init__()
        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            if in_channel == stage_ch:
                self.layers.append(nn.Sequential(
                    conv3x3(in_channel, stage_ch)))
            else:
                self.layers.append(nn.Sequential(
                    conv1x1(in_channel, stage_ch)))

            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            conv1x1(in_channel, concat_ch, 1))

        self.SE = SE
        if self.SE:
            self.ese = eSE(concat_ch)
        # self.cbam = CBAM(concat_ch)
        
    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.SE:
            xt = self.ese(xt)
        if self.identity:    
            xt = xt + identity_feat

        return xt



class CSP_OSA_Stage(nn.Module):

    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, isDown=False, isFocus=False):
        super().__init__()

        self.block_per_stage = block_per_stage

        self.isDown = isDown
        if self.isDown:
            if isFocus:
                self.downsample = Focus(in_ch, in_ch)
            else:
                self.downsample = nn.Sequential(
                    nn.MaxPool3d(2, 2)
                    # conv1x1(in_ch * 2, in_ch, 1)
                )


        m = [OSA_Module(in_ch, stage_ch, concat_ch, layer_per_block, False, False)]
        
        for i in range(block_per_stage - 1):
            m.append(OSA_Module(concat_ch, stage_ch, concat_ch, layer_per_block, True, i == block_per_stage - 2))
        self.m = nn.Sequential(*m)
        

    def forward(self, input):
        if self.isDown:
            input = self.downsample(input) 

        out = self.m(input)
    
        return out

class Focus(nn.Module):

    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = conv3x3(c1, c2, 1)
    def forward(self, input):
        l1 = self.conv(input[..., ::2, ::2, ::2]) + self.conv(input[..., 1::2, ::2, ::2]) + self.conv(input[..., ::2, 1::2, ::2]) + self.conv(input[..., ::2, ::2, 1::2]) + self.conv(input[..., 1::2, 1::2, ::2]) + self.conv(input[..., ::2, 1::2, 1::2]) + self.conv(input[..., 1::2, ::2, 1::2]) + self.conv(input[..., 1::2, 1::2, 1::2])
        l1 = l1 / 8
        return l1

class Upsample(nn.Module):

    def __init__(self, in_ch, out_ch, scale):
        super().__init__()
        self.conv = conv1x1(in_ch, out_ch, 1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, input):
        input = self.conv(input)
        return self.up(input)

class RFB(nn.Module):

    def __init__(self, plane, out):
        super().__init__()
        self.conv1x1 = conv1x1(plane, out)

        self.conv1x1_in = nn.Sequential(
            conv1x1(out, out),
            conv3x3(out, out, dilation=1)
        )
        
        self.conv3x3_in = nn.Sequential( 
            conv3x3(out, out),
            conv3x3(out, out, dilation=3)
        )
        self.conv5x5_in = nn.Sequential(
            nn.Conv3d(out, out, 5, 1, 2, bias=False),
            nn.BatchNorm3d(out),
            Mish(),
            conv3x3(out, out, dilation=5)
        )

        self.conv1x1_out = conv1x1(out * 3, out)

        self.ese = eSE(out)


    def forward(self, input):
        input = self.conv1x1(input)
        identity = input
        conv1x1_in = self.conv1x1_in(input)
        conv3x3_in = self.conv3x3_in(input)
        conv5x5_in = self.conv5x5_in(input)
        combine = torch.cat((conv1x1_in, conv3x3_in, conv5x5_in), 1)
        return self.ese(self.conv1x1_out(combine)) + identity

class VoVNet(nn.Module):

    def __init__(self, config_stage_ch, config_concat_ch, block_per_stage, layer_per_block):
        super(VoVNet, self).__init__()

        basic_ch = 64
        self.isFocus = False
        self.basic_conv = nn.Sequential(
            nn.Conv3d(1, basic_ch, 5, 1, 2, bias=False),
            nn.BatchNorm3d(basic_ch),
            Mish(),
            nn.MaxPool3d(2, 2)
        )

        basic_out = [basic_ch]
        in_ch_list = basic_out + config_concat_ch[:-1] 

        self.stage00 = CSP_OSA_Stage(64, 32, 64, 2, 8, isDown=False, isFocus=False)
        self.stage01 = CSP_OSA_Stage(64, 32, 64, 4, 8, isDown=True, isFocus=False)
        self.stage12 = CSP_OSA_Stage(64, 32, 64, 4, 8, isDown=True, isFocus=False)
        self.stage23 = CSP_OSA_Stage(64, 32, 64, 2, 8, isDown=True, isFocus=False)
        
        self.up1 = Upsample(64, 64, 2)
        self.rfb1 = RFB(128, 64)

        self.up2 = Upsample(64, 64, 2)
        self.rfb2 = RFB(128, 64)
        
        self.up3 = Upsample(64, 64, 2)
        self.rfb3 = RFB(128, 64)
        
        self.downsample = nn.MaxPool3d(2, 2)
        self.rfb4 = RFB(128, 64)

        self.head20 = nn.Conv3d(64, len(config['anchors']) * 5, 1 , 1, 0)

    def forward(self, input, coord, mode='train'):    
        recon= input

        x = self.basic_conv(input) # 128
        
        l00 = self.stage00(x)   # 40
        l01 = self.stage01(l00) # 20
        l12 = self.stage12(l01) # 10 
        l23 = self.stage23(l12) # 5

        l23_up = self.up1(l23)
        l12_combine = torch.cat((l23_up, l12), 1)
        l12_combine = F.dropout(l12_combine, 0.3)  if mode == 'train' else l12_combine
        l12_final = self.rfb1(l12_combine)

        l12_up = self.up2(l12_final)
        l01_combine = torch.cat((l12_up, l01), 1)
        l01_combine = F.dropout(l01_combine, 0.3)  if mode == 'train' else l01_combine
        l01_final = self.rfb2(l01_combine)

        l01_up = self.up3(l01_final)
        l00_combine = torch.cat((l01_up, l00), 1)
        l00_combine = F.dropout(l00_combine, 0.3) if mode == 'train' else l00_combine
        l00_final = self.rfb3(l00_combine)

        final = self.downsample(l00_final)
        final = torch.cat((final, l01_final), 1)
        final = F.dropout(final, 0.3) if mode == 'train' else final
        final = self.rfb4(final)

        cls_out20 = self.head20(final)
        cls_size = cls_out20.size()
        cls_out20 = cls_out20.view(cls_out20.size(0), cls_out20.size(1), -1)
        cls_out20 = cls_out20.transpose(1, 2).contiguous().view(cls_size[0], cls_size[2], cls_size[3], cls_size[4], len(config['anchors']), -1)
        cls_out20[..., 0] = torch.sigmoid(cls_out20[..., 0])


        return cls_out20, recon



def get_model(output_feature=False):
    
    net = VoVNet(
        config_stage_ch = [32, 32, 32, 32], \
        config_concat_ch= [64, 96, 128, 196], \
        block_per_stage = [1, 2, 2, 1], \
        layer_per_block = [3, 6, 8, 12], 
    )
    # print(net)
    # loss = FocalLoss(config['num_hard'])
    loss = Loss_recon(config['num_hard'], class_loss='BCELoss')

    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb



if __name__ == '__main__':


    net = VoVNet([32, 32, 32, 32], [64, 64, 64, 64], [1, 1, 2, 2], 5)
    print(net)
    x = torch.zeros((2, 1, 96, 96, 96))
    coord = torch.zeros(2, 3, 24, 24, 24)
    print(net(x, coord).size())








