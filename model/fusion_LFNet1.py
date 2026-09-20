import torch
import torch.nn as nn
class Fusion(nn.Module):
    """
    Saliency-guided skip fusion.

    The max-pooled input image is only used to generate a spatial gate
    for enhancing encoder skip features. It is not directly concatenated
    as an additional feature branch.
    """
    def __init__(self, in_high_channels, in_low_channels, out_channels, img_channels=1, prompt_channels=None):
        super().__init__()

        if prompt_channels is None:
            prompt_channels = max(4, out_channels // 4)

        self.img_proj = nn.Sequential(
            nn.Conv2d(img_channels, prompt_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU(inplace=True)
        )

        self.saliency_gate = nn.Sequential(
            nn.Conv2d(prompt_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
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
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_feat, low_feat, img_prompt):
        target_size = low_feat.shape[-2:]

        if high_feat.shape[-2:] != target_size:
            high_feat = F.interpolate(
                high_feat,
                size=target_size,
                mode='bilinear',
                align_corners=True
            )

        if img_prompt.shape[-2:] != target_size:
            img_prompt = F.adaptive_max_pool2d(img_prompt, target_size)

        img_p = self.img_proj(img_prompt)
        gate = self.saliency_gate(img_p)  # [B, 1, H, W]

        high = self.high_proj(high_feat)  # [B, C_out, H, W]
        low = self.low_proj(low_feat)     # [B, C_out, H, W]

        low = low * (1.0 + gate)

        fused = self.fuse(torch.cat([high, low], dim=1))

        return fused