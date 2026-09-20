import torch
import torch.nn as nn
from torch.ao.nn.quantized import BatchNorm2d


# DSA

class CONV(nn.Module):
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, d=1, act=True):
        super(CONV, self).__init__()

        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, dilation=d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Fuse(nn.Module):
    def __init__(self, in_gaussian_channel, in_low_channels, in_high_channels, out_channels=64, r=4):
        super(Fuse, self).__init__()
        # assert in_low_channels == out_channels

        self.in_gaussian_channel = in_gaussian_channel
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels

        self.bottleneck_channels = int(out_channels // r)

        self.feature_gaussian = nn.Sequential(
            nn.Conv2d(self.in_gaussian_channel, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )  ##512

        # self.feature_high = nn.Sequential(
        #     nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
        #     nn.BatchNorm2d(self.out_channels),
        #     nn.ReLU(True),
        # )  ##512

        self.feature_low = nn.Sequential(
            nn.Conv2d(self.in_gaussian_channel+in_low_channels, in_low_channels, 1, 1, 0),
            nn.BatchNorm2d(in_low_channels),
            nn.ReLU(True),
        )  ##512


        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.high_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.low_channels, 1, 1, 0),
            nn.BatchNorm2d(self.low_channels),
            nn.Sigmoid()
        )  # 512

        ##############add spatial attention ###Cross UtU############
        self.enspatial = SpatialAttention_l()
        self.despatial = SpatialAttention_h(in_low_channels,in_high_channels)



        self.post = nn.Sequential(
            nn.Conv2d(self.high_channels, self.high_channels, 3, 1, 1),
            nn.BatchNorm2d(self.high_channels),
            nn.ReLU(True),
        )  # 512

    def forward(self, xg, xl, xh):
        theta1 = self.enspatial(xg, xl)

        # xg = self.feature_gaussian(xg)
        # xl = self.feature_low(xl)
        # xh = self.feature_high(xh)

        x_low = self.feature_low(torch.cat([xg + theta1 * xg, xl + theta1 * xl],dim=1))


        w_t = self.topdown(xh)


        xl_tau = x_low * w_t

        theta2 = self.despatial(xl_tau, xh)
        out = self.post(theta2 * xh + (1 - theta2) * x_low)

        return out

        ##############################

class SpatialAttention_l(nn.Module):
    def __init__(self, kernel_size = 3):
        super(SpatialAttention_l, self).__init__()


        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.bn = nn.BatchNorm2d(1)
        self.conv = nn.Conv2d(4, 1, kernel_size, padding=padding)

    def forward(self, x1, x2):
        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        max_out1, _ = torch.max(x1, dim=1, keepdim=True)
        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)

        theta = torch.sigmoid(self.bn(self.conv(torch.cat([avg_out1, max_out1, avg_out2, max_out1], dim=1))))
        return theta


class SpatialAttention_h(nn.Module):
    def __init__(self, low_channels, high_channels):
        super(SpatialAttention_h, self).__init__()


        self.convl = nn.Conv2d(in_channels=low_channels+high_channels,out_channels=1,kernel_size=3,stride=1,padding=1)


    def forward(self, xl, xh):
        # attenl = self.convl(xl)
        # attenh = self.convh(xh)
        theta = torch.sigmoid(self.convl(torch.cat([xl,xh],dim=1)))

        return theta


if __name__ == "__main__":
    x = torch.randn(4, 4, 4, 4)
    y = torch.randn(4, 8, 4, 4)
    z = torch.randn(4, 8, 4, 4)
    demo = Fuse(4, 8, 8)
    out = demo(x, y, z)
    print(z.size())