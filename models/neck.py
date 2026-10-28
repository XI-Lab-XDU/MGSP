import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class Neck(nn.Module):
    r"""Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features. 

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU').
        num_outs (int, optional): Number of output feature maps. There
            would be extra_convs when num_outs larger than the length
            of in_channels.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 ):
        super(Neck, self).__init__()
        assert isinstance(in_channels, list)
        self.extra_convs = None
        self.convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        for i in range(len(in_channels)): # 为每个通道创造横向卷积和可变形卷积
            l_conv = ConvModule(
                in_channels=in_channels[i],
                out_channels=out_channels,
                kernel_size=1 if i == 2 else kernel_size,
                padding=0 if i == 2 else (kernel_size - 1) // 2,
                norm=True,
                activation=True)

            conv = DeformConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=2 if i == 0 else 1,
                norm=True,
                activation=True
            )
            self.lateral_convs.append(l_conv)
            self.convs.append(conv)


    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.convs) # 确保通道数一致
        # outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        outs = []
        prev_shape = inputs[-2].shape[2:] # 获取倒数第二个输入特征图的空间尺寸（高和宽），这将在后续用于调整特征图的尺寸
        for i in range(len(self.convs) - 1, -1, -1): # 逆序遍历卷积层
            lateral = self.lateral_convs[i](inputs[i]) # 对第i个特征进行横向卷积
            if i == len(self.convs) - 1:
                out = F.interpolate(lateral, size=prev_shape, mode='nearest') # 使用插值操作将横向卷积的输出lateral调整为与倒数第二个特征图相同的空间尺寸
            else:
                out = lateral
            out = self.convs[i](out) # 进行可变形卷积
            outs.append(out)
        return tuple(outs)
    
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        norm=True,
        activation=True,
        inplace=True
    ):
        super(ConvModule, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, bias=bias))

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation:
            layers.append(nn.ReLU(inplace=inplace))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class DeformConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False, norm=True, activation=True):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.act = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        if self.norm:
            out = self.norm(out)
        if self.act:
            out = self.act(out)
        return out