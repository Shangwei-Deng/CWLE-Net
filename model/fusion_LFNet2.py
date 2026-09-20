import torch
import torch.nn as nn
class Fusion(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, img_channels=1, prompt_channels=None):
        super().__init__()

        if prompt_channels is None:
            prompt_channels = max(4, out_channels // 4)

        self.img_proj = nn.Sequential(
            nn.Conv2d(img_channels, prompt_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(prompt_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.high_proj = nn.Sequential(
            nn.Conv2d(in_high_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.low_proj = nn.Sequential(
            nn.Conv2d(in_low_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_feat, low_feat, img_prompt):
        target_size = low_feat.shape[-2:]

        if high_feat.shape[-2:] != target_size:
            high_feat = F.interpolate(high_feat, size=target_size, mode='bilinear', align_corners=True)

        if img_prompt.shape[-2:] != target_size:
            img_prompt = F.adaptive_max_pool2d(img_prompt, target_size)

        high = self.high_proj(high_feat)
        low = self.low_proj(low_feat)
        prompt = self.img_proj(img_prompt)

        fused = self.fuse(torch.cat([high, low, prompt], dim=1))

        return fused