# Copyright (c) OpenMMLab. All rights reserved.
from PIL import Image
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmpretrain.models.utils import GRN, build_norm_layer
from mmpretrain.models.backbones.base_backbone import BaseBackbone

import torch.nn.functional as F
import pywt
import math


@MODELS.register_module()
class WTConvNeXt(BaseBackbone):
    """WTConvNeXt v1&v2 backbone.

    A PyTorch implementation of `A ConvNet for the 2020s
    <https://arxiv.org/abs/2201.03545>`_ and
    `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
    <http://arxiv.org/abs/2301.00808>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    To use WTConvNeXt v2, please set ``use_grn=True`` and ``layer_scale_init_value=0.``.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.
            - wt_levels (list[int]): The number of WTConv levels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        use_grn (bool): Whether to add Global Response Normalization in the
            blocks. Defaults to False.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        # 'atto': {
        #     'depths': [2, 2, 6, 2],
        #     'channels': [40, 80, 160, 320]
        # },
        # 'femto': {
        #     'depths': [2, 2, 6, 2],
        #     'channels': [48, 96, 192, 384]
        # },
        # 'pico': {
        #     'depths': [2, 2, 6, 2],
        #     'channels': [64, 128, 256, 512]
        # },
        # 'nano': {
        #     'depths': [2, 2, 8, 2],
        #     'channels': [80, 160, 320, 640]
        # },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768],
            'wt_levels': [5, 4, 3, 2]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768],
            'wt_levels': [5, 4, 3, 2]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024],
            'wt_levels': [5, 4, 3, 2]
        },
        # 'large': {
        #     'depths': [3, 3, 27, 3],
        #     'channels': [192, 384, 768, 1536]
        # },
        # 'xlarge': {
        #     'depths': [3, 3, 27, 3],
        #     'channels': [256, 512, 1024, 2048]
        # },
        # 'huge': {
        #     'depths': [3, 3, 27, 3],
        #     'channels': [352, 704, 1408, 2816]
        # }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        self.wt_levels = arch['wt_levels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels) == len(self.wt_levels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]
            wt_levels = self.wt_levels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                WTConvNeXtBlock(
                    in_channels=channels,
                    dw_conv_cfg=dict(kernel_size=5, stride=1, bias=True, wt_levels=wt_levels),
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(WTConvNeXt, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2


class WTConvNeXtBlock(BaseModule):
    """ConvNeXt Block.


    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise WTConv convolution.
            Defaults to ``dict(kernel_size=5, stride=1, bias=True, wt_levels=2)``.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back


        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=5, stride=1, bias=True, wt_levels=2),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = WTConv2d(
            in_channels, in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)

                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class WTConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 bias=True,
                 wt_levels=1,
                 wt_type='db1',
                 se_ll=True):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = _create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(_wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(_inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


def _create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def _wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def _inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class TemporalFusion(nn.Module):
    def __init__(self, in_channel, num_frame, norm_type='BN'):
        super(TemporalFusion, self).__init__()
        self.num_frame = num_frame
        self.conv_f = nn.Sequential(
            ConvModule(
                in_channels=in_channel * 2,
                out_channels=in_channel,
                kernel_size=1,
                norm_cfg=dict(type=norm_type),
                act_cfg=dict(type="Sigmoid"),
            ))
        self.conv_i = nn.Sequential(
            ConvModule(
                in_channels=in_channel * 2,
                out_channels=in_channel,
                kernel_size=1,
                norm_cfg=dict(type=norm_type),
                act_cfg=dict(type="Sigmoid"),
            ))

        self.conv_c_hat = nn.Sequential(
            ConvModule(
                in_channels=in_channel * 2,
                out_channels=in_channel,
                kernel_size=1,
                norm_cfg=dict(type=norm_type),
                act_cfg=dict(type="Tanh"),
            ))
        self.conv_o = nn.Sequential(
            ConvModule(
                in_channels=in_channel * 2,
                out_channels=in_channel,
                kernel_size=1,
                norm_cfg=dict(type=norm_type),
                act_cfg=dict(type="Sigmoid"),
            ))
        self.conv_c = nn.Sequential(
            ConvModule(
                in_channels=in_channel * 2,
                out_channels=in_channel,
                kernel_size=1,
                norm_cfg=dict(type=norm_type),
                act_cfg=dict(type="Tanh"),
            ))

    def forward(self, cur_x):
        hs = []
        c = torch.zeros_like(cur_x[0], device=cur_x[0].device, dtype=torch.float32)
        h = torch.zeros_like(cur_x[0], device=cur_x[0].device, dtype=torch.float32)
        # h = torch.zeros_like(cur_x[0])
        for frame in range(self.num_frame):
            f = self.conv_f(torch.cat([cur_x[frame], h], dim=1))
            i = self.conv_i(torch.cat([cur_x[frame], h], dim=1))
            c_hat = self.conv_c_hat(torch.cat([cur_x[frame], h], dim=1))
            o = self.conv_o(torch.cat([cur_x[frame], h], dim=1))
            c = self.conv_c(torch.cat([f * c, i * c_hat], dim=1))
            h = o * F.leaky_relu(c)
            hs.append(h)
        return hs


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list
        # assert mask is not None
        mask = torch.zeros_like(x[:,0,:,:], dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_emb_cur = PositionEmbeddingSine(num_pos_feats=dim // 2, normalize=True)
        self.pos_emb_memory = PositionEmbeddingSine(num_pos_feats=dim // 2, normalize=True)


    def forward(self, cur_frame, memory, h_cur, h_memory):
        b, c, _= cur_frame.shape
        cur_frame = cur_frame.reshape(b, c, h_cur, -1)
        memory = memory.reshape(b, c, h_memory, -1)
        cur_frame = cur_frame + self.pos_emb_cur(cur_frame)
        memory = memory + self.pos_emb_memory(memory)

        if cur_frame.dim() == 4:
            b, c, w, h = cur_frame.shape
            cur_frame = cur_frame.view(b, c, h * w)
        if memory.dim() == 4:
            b, c, w, h = memory.shape
            memory = memory.view(b, c, h * w)

        cur_frame = cur_frame.permute(0, 2, 1)
        memory = memory.permute(0, 2, 1)

        B, L1, C = cur_frame.shape
        B, L2, C = memory.shape

        q = self.q_proj(cur_frame)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        q = q.reshape(B, L1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, L2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, L2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.permute(0, 2, 1)

        return out


class LSMemory(nn.Module):
    def __init__(self,
                 channel=None):
        super(LSMemory, self).__init__()
        self.long_memory = []
        self.short_memory = []
        self.long_term_attention = CrossAttention(dim=channel, qkv_bias=True)
        self.short_term_attention = CrossAttention(dim=channel, qkv_bias=True)
        self.ffn1 = nn.Conv2d(in_channels=channel, out_channels=channel*4, kernel_size=1)
        self.ffn2 = nn.Conv2d(in_channels=channel*4, out_channels=channel, kernel_size=1)
        self.conv_long_short = nn.Conv2d(in_channels=2*channel, out_channels=channel, kernel_size=1)

    def layer_norm(self, x):
        b, c, h, w = x.size()
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        normalized_shape = [c]
        x_norm = F.layer_norm(x_reshaped, normalized_shape)
        x_norm = x_norm.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x_norm

    def forward(self, x, frame_num=None):
        if 0 in frame_num:
            residual = x
            x = self.layer_norm(x)
            self.init_memory(x, frame_num)
            return residual
        else:
            x_pre = x
            x = self.layer_norm(x)
            long_term_output = self.long_term(x)
            short_term_output = self.short_term(x)
            self.update_memory(x)
            x = self.conv_long_short(torch.cat([long_term_output, short_term_output], dim=1)) + x_pre
            residual = x
            x = self.layer_norm(x)
            x = self.ffn1(x)
            x = F.gelu(x)
            x = self.ffn2(x)
            output = x + residual
            return output

    def init_memory(self, input_x, frame_num):
        b, c, _, _ = input_x.shape
        index = frame_num.index(0)
        self.long_memory = []
        self.short_memory = []
        for i in range(index, b):
            self.long_memory.append(input_x[i, ...].view(1, c, -1).clone().detach())
            self.short_memory.append(input_x[i, ...].view(1, c, -1).clone().detach())

    def long_term(self, cur_frame):
        num = len(self.long_memory)
        long_memory = torch.cat(self.long_memory, dim=0).transpose(0,1)
        b, c, h, w = cur_frame.shape
        long_memory = long_memory.contiguous().view(1, c, -1).expand(b, -1, -1)
        cur_frame = cur_frame.reshape(b, c, h * w)
        output = self.long_term_attention(cur_frame, long_memory, h, num * h)
        output = output.reshape(b, c, h, w)
        return output

    def short_term(self, cur_frame):
        num = len(self.short_memory)
        short_memory = torch.cat(self.short_memory, dim=0).transpose(0,1)
        b, c, h, w = cur_frame.shape
        short_memory = short_memory.contiguous().view(1, c, -1).expand(b, -1, -1)
        cur_frame = cur_frame.reshape(b, c, h * w)
        # short_memory = self.short_memory[-1]
        output = self.short_term_attention(cur_frame, short_memory, h, num * h)
        output = output.reshape(b, c, h, w)
        return output

    def update_memory(self, input_x):
        b, c, h, w = input_x.shape
        dev = input_x.device
        if(len(self.short_memory) < 2):
            self.short_memory.append(input_x[-1, ...].view(1, c, -1).clone().detach())
        else:
            self.short_memory.clear()
            if(b > 1):
                self.short_memory.append(input_x[-2, ...].view(1, c, -1).clone().detach())
                self.short_memory.append(input_x[-1, ...].view(1, c, -1).clone().detach())
            else:
                self.short_memory.append(input_x[-1, ...].view(1, c, -1).clone().detach())
        if(len(self.long_memory) + b <= 5):
            for i in range(b):
                self.long_memory.append(input_x[i, ...].view(1, c, -1).clone().detach())
        else:
            dim_num = len(self.long_memory) + b
            compress_num = dim_num - 5
            for i in range(b):
                self.long_memory.append(input_x[i, ...].view(1, c, -1).clone().detach())
            long_mem = torch.cat(self.long_memory, dim=0).view(dim_num, -1)

            long_mem = F.normalize(long_mem, p=2, dim=-1, eps=1e-8)
            sim = torch.matmul(long_mem, long_mem.T)
            sim[torch.triu(torch.ones_like(sim), diagonal=0).bool()] = -1

            values, max_id = torch.topk(sim.view(-1), k=compress_num)
            rows = max_id // sim.shape[1]
            cols = max_id % sim.shape[1]
            if(torch.unique(rows).shape == rows.shape):
                select, _ = torch.sort(rows, descending=True)
                for i in select:
                    del self.long_memory[i]
            else:
                select, _ = torch.sort(cols, descending=True)
                for i in select:
                    del self.long_memory[i]


@MODELS.register_module()
class MWConv(WTConvNeXt):
    def __init__(self, frames=10, layer_fuse=False, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames
        temporal = []
        for i in range(self.num_stages):
            temporal.append(TemporalFusion(in_channel=self.channels[i], num_frame=self.frames))
        self.temporal = nn.ModuleList(temporal)
        self.layer_fuse = layer_fuse
        self.LSMemory = LSMemory(channel=self.channels[-1])
        self.last_number = 0

    def forward(self, input, number):
        if (self.last_number + 1 != number[0] and 0 not in number):
            self.last_number = number[-1]
            number[0] = 0
        else:
            self.last_number = number[-1]
        outs = []
        xs = torch.split(input, 3, dim=1)
        for i, stage in enumerate(self.stages):
            cur_x = []
            for x in xs:
                x = self.downsample_layers[i](x)
                x = stage(x)
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                cur_x.append(x)
            if not self.layer_fuse:
                xs = cur_x

            if (i == 3):
                if 0 in number:
                    cur_x = self.temporal[i](cur_x[::-1])
                    cur_x[-1] = self.LSMemory(cur_x[-1], number)
                else:
                    cur_x[-1] = self.LSMemory(cur_x[-1], number)
            else:
                cur_x = self.temporal[i](cur_x[::-1])
            if self.gap_before_final_norm:
                gap = cur_x[-1].mean([-2, -1], keepdim=True)
                outs.append(norm_layer(gap).flatten(1))
            else:
                outs.append(norm_layer(cur_x[-1]))
            if self.layer_fuse:
                xs = cur_x[::-1]
        return tuple(outs)


def save_featuremap(tensor, save_path):
    feature_map = tensor[0,0,:,:]
    feature_map = feature_map.unsqueeze(0).unsqueeze(0)
    feature_map = torch.nn.functional.interpolate(feature_map,
                                    size=(512,512),
                                    mode='bilinear')
    feature_map = feature_map.squeeze(0).squeeze(0)
    min_val = feature_map.min()
    max_val = feature_map.max()
    eps = 1e-8
    normalized = (feature_map - min_val) / (max_val - min_val + eps)
    gray_tensor = (normalized * 255).byte()
    gray_image = Image.fromarray(gray_tensor.cpu().numpy(), mode='L')
    gray_image.save(save_path)
