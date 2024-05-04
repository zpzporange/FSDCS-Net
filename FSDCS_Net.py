# Feature-Stable and DCT-Channel-Separated Neural Network
import torch
import torch.nn as nn
from torch.nn import init


# from torchsummary import summary


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


# 深度逐通道卷积
class DepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_size, padding=0):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1,
                                   padding=padding, groups=channels)

    def forward(self, x):
        x = self.depthwise(x)
        return x


# 高效通道注意力ECA模块
class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.ws = WS_Block()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, out, x):
        ws = self.ws(out, x)

        # feature descriptor on the global spatial information
        y = self.avg_pool(ws)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y.expand_as(x)


# 三相之力：通道注意力、深度逐通道卷积、残差连接
class ECA_DC_ResBlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECA_DC_ResBlock, self).__init__()
        self.depthwise = DepthwiseConv(channels=channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels, momentum=0.1)
        self.prelu = nn.PReLU(num_parameters=channels)
        self.eca = eca_layer(channels, k_size)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.prelu(out)

        out = self.depthwise(out)
        weight = self.eca(out, x)

        out = self.depthwise(weight * (out + x))
        out = self.bn(out)
        out = self.prelu(out)

        return out


# 循环残差神经网络模块
class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = ECA_DC_ResBlock(channels=ch_out)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.1)
        self.prelu = nn.PReLU(num_parameters=ch_out)
        # 添加一个小于1的随机初始化的可学习权重
        self.residual_weight = nn.Parameter(torch.rand(1, 64, 64, 64) * 0.5)  # 随机初始化权重范围 [0, 0.5]

    def forward(self, x):
        # 确保在前向传播时权重不会超过 [-0.5, 0.5] 范围
        residual_weight = torch.clamp(self.residual_weight, min=-0.5, max=0.5)

        x1 = self.conv(x)
        for i in range(1, self.t):
            x1 = self.conv(x * residual_weight + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t - 1),
            Recurrent_block(ch_out, t=t)
        )

    def forward(self, x):
        return self.RCNN(x)


# 自定义加权残差
def flip_last_bit(tensor):
    # 将张量转换为整数
    tensor = tensor.to(torch.int64)

    # 创建一个与输入张量相同形状的张量，其所有元素都是1（二进制的最后一位）
    last_bits = torch.ones_like(tensor)

    # 使用异或操作翻转最后一位
    flipped = tensor ^ last_bits

    # 对于负数，我们需要将其转换回补码表示
    mask = tensor < 0  # 创建一个掩码，标记负数位置
    flipped[mask] = -(~(flipped[mask]) + 1)  # 将负数的非符号位取反并加1

    return flipped


class WS_Block(nn.Module):
    def __init__(self):
        super(WS_Block, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, 64, 64, 64))  # Initialize weight to random values
        self.bias = nn.Parameter(torch.zeros(1, 64, 64, 64))  # Initialize bias to zero

    def forward(self, x, xo):
        residual = (xo - x) * (xo - flip_last_bit(xo))
        # Apply the learned weight and bias with clamping
        x = torch.clamp(self.weight, -0.5, 0.5) * residual + torch.clamp(self.bias, -1, 1) + x
        return x


# 网络模型
class FSDCS_Net(nn.Module):
    def __init__(self, batch_size=1, t=2):
        super(FSDCS_Net, self).__init__()

        self.RRCNN1 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)
        self.RRCNN6 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)
        self.RRCNN7 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)
        self.RRCNN8 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)
        self.RRCNN9 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)
        self.RRCNN10 = RRCNN_block(ch_in=batch_size * 64, ch_out=batch_size * 64, t=t)

        self.WS1 = WS_Block()
        self.WS2 = WS_Block()
        self.WS3 = WS_Block()
        self.WS4 = WS_Block()
        self.WS5 = WS_Block()

    def forward(self, x):
        x1 = self.RRCNN1(x)
        x1 = self.RRCNN2(x1)
        ws1 = self.WS1(x1, x)

        x2 = self.RRCNN3(ws1)
        x2 = self.RRCNN4(x2)
        ws2 = self.WS2(x2, x)

        x3 = self.RRCNN5(ws2)
        x3 = self.RRCNN6(x3)
        ws3 = self.WS3(x3, x)

        x4 = self.RRCNN7(ws3)
        x4 = self.RRCNN8(x4)
        ws4 = self.WS4(x4, x)

        x5 = self.RRCNN9(ws4)
        x5 = self.RRCNN10(x5)
        out = self.WS5(x5, x) - x

        return out

# """print layers and params of network"""
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = FSDCS_Net().to(device)
#     summary(model, (64, 64, 64))
