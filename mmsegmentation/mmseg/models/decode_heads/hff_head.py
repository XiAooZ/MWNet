import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple

from mmcv.cnn import ConvModule
from .decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from functools import partial
from mmengine.model import BaseModule
from ..backbones.MWConv import _create_wavelet_filter, _wavelet_transform, _inverse_wavelet_transform

HORIZONTAL_FIRST = True


class Splitting(BaseModule):
    def __init__(self, horizontal):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        if (horizontal):
            self.conv_even = lambda x: x[:, :, :, ::2]
            self.conv_odd = lambda x: x[:, :, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, :, ::2, :]
            self.conv_odd = lambda x: x[:, :, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))


class SplittingInverse(BaseModule):
    def __init__(self, horizontal):
        super().__init__()
        self.horizontal = horizontal

    def forward(self, even, odd):
        b, c, h, w = even.size()
        h = max(h, w)
        if self.horizontal:
            x = torch.repeat_interleave(even, 2, dim=-1)
            x[:, :, :, 1::2] = odd
        else:
            x = torch.repeat_interleave(even, 2, dim=-2)
            x[:, :, 1::2, :] = odd
        return x


class AWT1D(BaseModule):
    def __init__(self, horizontal, in_planes, modified=True, splitting=True, k_size=4, size_hidden=2):
        super(AWT1D, self).__init__()
        self.modified = modified
        if horizontal:
            kernel_size = (1, k_size)
            pad = (k_size // 2, k_size - 1 - k_size // 2, 0, 0)
        else:
            kernel_size = (k_size, 1)
            pad = (0, 0, k_size // 2, k_size - 1 - k_size // 2)
        self.SplittingInverse = SplittingInverse(horizontal)

        self.splitting = splitting
        self.split = Splitting(horizontal)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1

        modules_P+= [
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_planes * prev_size, in_planes * size_hidden,
                      kernel_size=kernel_size, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_planes * size_hidden, in_planes,
                      kernel_size=(1, 1), stride=1),
            nn.Tanh()]
        modules_U+= [
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_planes * prev_size, in_planes * size_hidden,
                      kernel_size=kernel_size, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_planes * size_hidden, in_planes,
                      kernel_size=(1, 1), stride=1),
            nn.Tanh()]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if not self.modified:
            x_even = x_even + self.U(x_odd)
            x_odd = x_odd - self.P(x_even)
        else:
            x_odd = x_odd - self.P(x_even)
            x_even = x_even + self.U(x_odd)
        return (x_even, x_odd)

    def inverse(self, x_even, x_odd):
        if self.modified:
            x_even = x_even - self.U(x_odd)
            x_odd = x_odd + self.P(x_even)
        else:
            x_odd = x_odd + self.P(x_even)
            x_even = x_even - self.U(x_odd)
        x = self.SplittingInverse(x_even, x_odd)
        return x


class AWT2D(BaseModule):
    def __init__(self, in_planes, share_weights=False, modified=True, kernel_size=4):
        super(AWT2D, self).__init__()
        self.level1_lf = AWT1D(
            horizontal=HORIZONTAL_FIRST, in_planes=in_planes, modified=modified,
            k_size=kernel_size)
        if share_weights:
            self.level2_1_lf = AWT1D(
                horizontal=not HORIZONTAL_FIRST, in_planes=in_planes, modified=modified,
                k_size=kernel_size)
            self.level2_2_lf = self.level2_1_lf  # Double check this
        else:
            self.level2_1_lf = AWT1D(
                horizontal=not HORIZONTAL_FIRST, in_planes=in_planes, modified=modified,
                k_size=kernel_size)
            self.level2_2_lf = AWT1D(
                horizontal=not HORIZONTAL_FIRST, in_planes=in_planes, modified=modified,
                k_size=kernel_size)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.level1_lf(x)
        (LL, LH) = self.level2_1_lf(c)
        (HL, HH) = self.level2_2_lf(d)

        return (LL, LH, HL, HH)

    def inverse(self, LL, LH, HL, HH):
        c = self.level2_1_lf.inverse(LL, LH)
        d = self.level2_2_lf.inverse(HL, HH)
        x = self.level1_lf.inverse(c, d)
        return x


class ChannelAttention(BaseModule):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class BasicWT(BaseModule):
    def __init__(self, in_channels, wt_type='haar', **kwargs):
        super(BasicWT, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.wt_filter, self.iwt_filter = _create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(_wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(_inverse_wavelet_transform, filters=self.iwt_filter)

    def forward(self, x):
        wt_x = self.wt_function(x)
        return wt_x[:, :, 0, :, :], wt_x[:, :, 1, :, :], wt_x[:, :, 2, :, :], wt_x[:, :, 3, :, :]

    def inverse(self, LL, LH, HL, HH):
        wt_x = torch.stack((LL, LH, HL, HH), dim=2)
        return self.iwt_function(wt_x)


class WaveletFilter(BaseModule):
    def __init__(self, in_channels, wt_type='adaptive', share_weights=False, kernel_size=3, simple_lifting=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        if wt_type == 'adaptive':
            self.AWF = AWT2D(
                in_planes=in_channels,
                kernel_size=kernel_size)
        else:
            self.AWF = BasicWT(
                in_channels=in_channels,
                wt_type=wt_type)
        self.CA = ChannelAttention(num_feat=in_channels * 4)

    def forward(self, x, upsample=False):
        LL, LH, HL, HH = self.AWF(x)
        wavelet_feature = torch.cat((LL, LH, HL, HH), dim=1)
        if upsample:
            h = upsample[0] // 2
            w = upsample[1] // 2
            wavelet_feature = F.interpolate(wavelet_feature, size=[h, w], mode='bilinear')
        residual = wavelet_feature
        wavelet_feature = self.CA(wavelet_feature)
        wavelet_feature = wavelet_feature + residual
        LL = wavelet_feature[:, :self.in_channels, :, :]
        LH = wavelet_feature[:, self.in_channels:self.in_channels * 2, :, :]
        HL = wavelet_feature[:, self.in_channels * 2:self.in_channels * 3, :, :]
        HH = wavelet_feature[:, self.in_channels * 3:self.in_channels * 4, :, :]
        x = self.AWF.inverse(LL, LH, HL, HH)

        return x


class HFF(BaseModule):
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 compressed_channels=64,
                 share_weights=False,
                 need_ctrans=True,
                 wt_type='adaptive',
                 wavekernel_size=4,
                 simple_lifting=False,
                 initial_fusion=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.initial_fusion = initial_fusion
        self.wavefilter_lr = WaveletFilter(in_channels=compressed_channels,
                                           share_weights=share_weights,
                                           kernel_size=wavekernel_size,
                                           wt_type=wt_type,
                                           simple_lifting=simple_lifting)
        self.wavefilter_hr = WaveletFilter(in_channels=compressed_channels,
                                           share_weights=share_weights,
                                           kernel_size=wavekernel_size,
                                           wt_type=wt_type,
                                           simple_lifting=simple_lifting)
        if initial_fusion:
            self.wavefilter_lr1 = WaveletFilter(in_channels=compressed_channels,
                                                share_weights=share_weights,
                                                kernel_size=wavekernel_size,
                                                wt_type=wt_type,
                                                simple_lifting=simple_lifting)
            self.wavefilter_hr1 = WaveletFilter(in_channels=compressed_channels,
                                                share_weights=share_weights,
                                                kernel_size=wavekernel_size,
                                                wt_type=wt_type,
                                                simple_lifting=simple_lifting)

        self.need_ctrans = need_ctrans
        if need_ctrans:
            self.transform_channels = nn.Conv2d(compressed_channels, hr_channels, kernel_size=1, stride=1)
            self.transform_channels1 = nn.Conv2d(compressed_channels, lr_channels, kernel_size=1, stride=1)

    def forward(self, hr_feat, lr_feat):
        """Fusion forward process

        Args:
            hr_feat: high resolution feature map in fusion
            lr_feat: low resolution feature map in fusion

        Returns:
            fusion result with high resolution
        """
        return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        pri_hr_feat = compressed_hr_feat + self.wavefilter_hr(compressed_hr_feat)
        pri_lr_feat = self.wavefilter_lr(compressed_lr_feat, compressed_hr_feat.shape[2:])
        if not self.initial_fusion:
            if self.need_ctrans:
                hr_feat = self.transform_channels(pri_hr_feat)
                lr_feat = self.transform_channels1(pri_lr_feat)
            return hr_feat, lr_feat
        pri_feat = pri_hr_feat + pri_lr_feat
        hr_feat = self.wavefilter_hr1(pri_feat) + compressed_hr_feat
        lr_feat = self.wavefilter_lr1(pri_feat)
        if self.need_ctrans:
            hr_feat = self.transform_channels(hr_feat)
            lr_feat = self.transform_channels1(lr_feat)
        return hr_feat, lr_feat


@MODELS.register_module()
class FPNDecoder(BaseDecodeHead):
    def __init__(self, conv_kernel=1, up_sample='nearest', **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.in_channels, list):
            self.num_feats = len(self.in_channels)
        else:
            self.num_feats = 1
        conv_module = []
        for feat in range(self.num_feats):
            conv_module.append(
                ConvModule(
                    in_channels=self.in_channels[feat],
                    out_channels=self.channels,
                    kernel_size=conv_kernel,
                    padding=conv_kernel // 2,
                    stride=1,
                    norm_cfg=dict(type='BN')
                )
            )
        self.up_sample = up_sample
        self.conv_modules = nn.ModuleList(conv_module)

    def _init_inputs(self, in_channels, in_index, input_transform):
        self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        input = []
        for index in self.in_index:
            input.append(inputs[index])
        return input

    def _fusion(self, x):
        for feat in range(self.num_feats):
            x[feat] = self.conv_modules[feat](x[feat])
        hr, lr = x[2], x[3]
        lr = F.interpolate(lr, size=hr.shape[-2:], mode=self.up_sample)
        lr = lr + hr
        hr = x[1]
        lr = F.interpolate(lr, size=hr.shape[-2:], mode=self.up_sample)
        lr = lr + hr
        hr = x[0]
        lr = F.interpolate(lr, size=hr.shape[-2:], mode=self.up_sample)
        output = lr + hr
        return output

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        fus_feat = self._fusion(x)
        return fus_feat

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class HFFDecoder(FPNDecoder):
    def __init__(self,
                 compress_ratio=8,
                 shared_weights=True,
                 wt_type='adaptive',
                 wavekernel_size=4,
                 initial_fusion=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.shared_weights = shared_weights
        if not shared_weights:
            fusion_modules = []
            for i in range(self.num_feats - 1):
                fusion_modules.append(HFF(
                    hr_channels=self.channels,
                    lr_channels=self.channels,
                    wt_type=wt_type,
                    compressed_channels=self.channels * 2 // compress_ratio,
                    wavekernel_size=wavekernel_size,
                    initial_fusion=initial_fusion
                ))
            self.waveletfusion_module = nn.ModuleList(fusion_modules)
        else:
            self.waveletfusion_module = HFF(
                hr_channels=self.channels,
                lr_channels=self.channels,
                compressed_channels=self.channels * 2 // compress_ratio,
                wt_type=wt_type,
                wavekernel_size=wavekernel_size,
                initial_fusion=initial_fusion
            )

    def _fusion(self, x):
        for feat in range(self.num_feats):
            x[feat] = self.conv_modules[feat](x[feat])
        if not self.shared_weights:
            output = x[-1]
            for i, waveletfusion_module in enumerate(self.waveletfusion_module):
                hr, lr = waveletfusion_module(hr_feat=x[-(i + 2)], lr_feat=output)
                output = hr + lr
        else:
            hr, lr = self.waveletfusion_module(hr_feat=x[2], lr_feat=x[3])
            output = hr + lr
            hr, lr = self.waveletfusion_module(hr_feat=x[1], lr_feat=output)
            output = hr + lr
            hr, lr = self.waveletfusion_module(hr_feat=x[0], lr_feat=output)
            output = hr + lr
        return output
