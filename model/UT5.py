import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from model.fusion_LFNet2 import Fusion as SaliencyGateSkipFusion
# from fusion_UIU import Fuse as SaliencyGateSkipFusion


try:
    from mamba_ssm import Mamba as MambaSSM
except Exception:
    MambaSSM = None
#############################################
#CWLE-Net 纯净版

from thop import profile


##################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, BasicConv=BasicConv):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)  # 这个也可以改成深度可分离

        self.dwconv = BasicConv(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, bias=bias,
                                relu=False, groups=hidden_features * 2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MambaBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, BasicConv=BasicConv):
        super(MambaBlock, self).__init__()
        if MambaSSM is None:
            raise ImportError(
                "mamba_ssm is required for MambaBlock. "
                "Please install mamba_ssm before running this model."
            )

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # Keep d_model fixed to channel dim; sequence length is HW at runtime.
        # This avoids huge parameter allocation when H*W is large.
        self.mamba = MambaSSM(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, BasicConv=BasicConv)

    def forward(self, x):
        b, c, h, w = x.shape
        # Full-resolution sequence modeling without spatial compression:
        # [B, C, H, W] -> [B, HW, C] -> Mamba -> [B, C, H, W]
        x_norm = self.norm1(x)
        mamba_in = to_3d(x_norm)  # [B, HW, C]
        mamba_out = self.mamba(mamba_in)  # [B, HW, C]
        mamba_out = to_4d(mamba_out, h, w)  # [B, C, H, W]
        x = x + mamba_out
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.GELU()
        self.relu_s1 = nn.ReLU(inplace=True)
        # self.relu_s1 = nn.SiLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)  # b,c,h,w->b,c/2,h,w->b,2c,h/2,w/2


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)  # b,c,h,w->b,2c,h,w->b,c/2,2h,2w


def _upsample_like(src, tar):
    _, _, hei, wid = tar.shape
    src = F.interpolate(src, size=[hei, wid], mode='bilinear', align_corners=True)

    return src


def _haar_dwt2(x):
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (-x00 - x01 + x10 + x11) * 0.5
    hl = (-x00 + x01 - x10 + x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5

    return ll, lh, hl, hh


def _haar_iwt2(ll, lh, hl, hh):
    x00 = (ll - lh - hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll + lh + hl + hh) * 0.5

    b, c, h, w = ll.shape
    out = torch.zeros(
        (b, c, h * 2, w * 2),
        dtype=ll.dtype,
        device=ll.device
    )

    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11

    return out


class DSConvGN(nn.Module):
    """
    Depthwise separable convolution + GroupNorm + GELU.
    Shared by WLBM and CSLR.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=False
        )

        self.pw = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            bias=False
        )

        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DirectionalBranch(nn.Module):
    """
    Direction-aware branch using anisotropic depthwise convolutions.
    Shared by WLBM and CSLR.
    """

    def __init__(self, ch):
        super().__init__()

        self.horizontal = nn.Sequential(
            nn.Conv2d(
                ch,
                ch,
                kernel_size=(1, 5),
                padding=(0, 2),
                groups=ch,
                bias=False
            ),
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, ch),
            nn.GELU()
        )

        self.vertical = nn.Sequential(
            nn.Conv2d(
                ch,
                ch,
                kernel_size=(5, 1),
                padding=(2, 0),
                groups=ch,
                bias=False
            ),
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.horizontal(x) + self.vertical(x)


class LorentzEmbedding(nn.Module):
    """
    Decode Lorentz feature into new LL/LH/HL bands.

    Input:
        z_lor: [B, D, H/2, W/2]

    Output:
        new_ll, new_lh, new_hl:
        each [B, C, H/2, W/2]
    """

    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_ch),
            nn.GELU()
        )

        self.branch_local = DSConvGN(hidden_ch, hidden_ch, kernel_size=3, padding=1, dilation=1)
        self.branch_mid = DSConvGN(hidden_ch, hidden_ch, kernel_size=3, padding=2, dilation=2)
        self.branch_large = DSConvGN(hidden_ch, hidden_ch, kernel_size=3, padding=3, dilation=3)
        self.branch_dir = DirectionalBranch(hidden_ch)

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_ch * 4, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_ch),
            nn.GELU()
        )

        self.out = nn.Conv2d(hidden_ch, out_ch * 3, kernel_size=1, bias=True)

    def forward(self, z_lor):
        x = self.stem(z_lor)

        f_local = self.branch_local(x)
        f_mid = self.branch_mid(x)
        f_large = self.branch_large(x)
        f_dir = self.branch_dir(x)

        feat = torch.cat([f_local, f_mid, f_large, f_dir], dim=1)
        feat = self.fuse(feat)

        struct_bands = self.out(feat)
        new_ll, new_lh, new_hl = torch.chunk(struct_bands, chunks=3, dim=1)

        return new_ll, new_lh, new_hl


class WLBM(nn.Module):
    """
    Wavelet-guided Lorentz Bottleneck Module.

    Single-stage version:
        LL/LH/HL -> Lorentz -> generate new LL/LH/HL
        keep HH unchanged
        IWT(LL', LH', HL', HH)
    """

    def __init__(self, dim, hidden_dim=None, eps=1e-6, lorentz_radius=1.0):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(16, dim // 2)
        assert self.hidden_dim >= 2, "hidden_dim must be at least 2 for [time, space]."

        self.space_dim = self.hidden_dim - 1
        self.eps = eps
        self.lorentz_radius = lorentz_radius

        self.band_proj = nn.Sequential(
            nn.Conv2d(dim * 3, self.space_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.space_dim),
            nn.GELU()
        )

        self.context_decoder = LorentzEmbedding(
            in_ch=self.hidden_dim,
            hidden_ch=self.hidden_dim,
            out_ch=dim
        )

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        nn.init.dirac_(self.out_proj.weight)

    def _dwt2_haar(self, x):
        return _haar_dwt2(x)

    def _iwt2_haar(self, ll, lh, hl, hh):
        return _haar_iwt2(ll, lh, hl, hh)

    def _to_lorentz_map(self, space):
        """
        Input:
            space: [B, D-1, H, W]

        Output:
            z_lor: [B, D, H, W] = [time, space]
        """
        space = torch.tanh(space) * self.lorentz_radius

        norm2 = torch.sum(space * space, dim=1, keepdim=True)
        time = torch.sqrt(torch.clamp(1.0 + norm2, min=self.eps))

        z_lor = torch.cat([time, space], dim=1)
        return z_lor

    def forward(self, x):
        b, c, h, w = x.shape

        if h < 4 or w < 4:
            return x

        pad_h = h % 2
        pad_w = w % 2

        if pad_h != 0 or pad_w != 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_pad = x

        ll, lh, hl, hh = self._dwt2_haar(x_pad)

        band_feat = torch.cat([ll, lh, hl], dim=1)
        space_embed = self.band_proj(band_feat)

        z_lor = self._to_lorentz_map(space_embed)

        new_ll, new_lh, new_hl = self.context_decoder(z_lor)

        rec = self._iwt2_haar(new_ll, new_lh, new_hl, hh)
        rec = rec[:, :, :h, :w]

        out = self.out_proj(rec)

        return out


class LLPromptExtractor(nn.Module):
    """
    Extract LL-only structural prompt from stage1/stage2.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU()
        )

    def _dwt2_haar(self, x):
        return _haar_dwt2(x)

    def forward(self, x, target_size):
        _, _, h, w = x.shape

        pad_h = h % 2
        pad_w = w % 2

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        ll, _, _, _ = self._dwt2_haar(x)

        ll = self.proj(ll)
        ll = F.adaptive_avg_pool2d(ll, target_size)

        return ll


class CrossStageStructuralLorentzRefinement(nn.Module):
    """
    Cross-stage Structural Lorentz Refinement, direct band generation version.

    Inputs:
        x3: stage3 feature
        x2: stage2 feature
        x1: stage1 feature

    Design:
        stage1/stage2 only provide LL prompts.
        stage3 provides LL3/LH3/HL3.
        HH3 is not used by Lorentz branch and is kept unchanged.
        Decoder directly generates new LL3/LH3/HL3.
    """

    def __init__(self, c3, c2, c1, hidden_dim=None, eps=1e-6, lorentz_radius=1.0):
        super().__init__()

        self.c3 = c3
        self.c2 = c2
        self.c1 = c1
        self.eps = eps
        self.lorentz_radius = lorentz_radius

        self.hidden_dim = hidden_dim if hidden_dim is not None else max(16, c3 // 2)
        assert self.hidden_dim >= 2, "hidden_dim must be at least 2."

        self.space_dim = self.hidden_dim - 1

        self.stage1_ll_prompt = LLPromptExtractor(c1, c3)
        self.stage2_ll_prompt = LLPromptExtractor(c2, c3)

        self.struct_proj = nn.Sequential(
            nn.Conv2d(c3 * 5, self.space_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.space_dim),
            nn.GELU()
        )

        self.struct_decoder = LorentzEmbedding(
            in_ch=self.hidden_dim,
            hidden_ch=self.hidden_dim,
            out_ch=c3
        )

        self.out_proj = nn.Conv2d(c3, c3, kernel_size=1, bias=False)
        nn.init.dirac_(self.out_proj.weight)

    def _dwt2_haar(self, x):
        return _haar_dwt2(x)

    def _iwt2_haar(self, ll, lh, hl, hh):
        return _haar_iwt2(ll, lh, hl, hh)

    def _to_lorentz_map(self, space):
        """
        Input:
            space: [B, D-1, H, W]

        Output:
            z_lor: [B, D, H, W] = [time, space]
        """
        space = torch.tanh(space) * self.lorentz_radius

        norm2 = torch.sum(space * space, dim=1, keepdim=True)
        time = torch.sqrt(torch.clamp(1.0 + norm2, min=self.eps))

        z_lor = torch.cat([time, space], dim=1)
        return z_lor

    def forward(self, x3, x2, x1):
        b, c, h, w = x3.shape

        if h < 8 or w < 8:
            return x3

        pad_h = h % 2
        pad_w = w % 2

        if pad_h != 0 or pad_w != 0:
            x3_pad = F.pad(x3, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x3_pad = x3

        ll3, lh3, hl3, hh3 = self._dwt2_haar(x3_pad)
        target_size = ll3.shape[-2:]

        ll1_prompt = self.stage1_ll_prompt(x1, target_size)
        ll2_prompt = self.stage2_ll_prompt(x2, target_size)

        struct_feat = torch.cat(
            [ll3, lh3, hl3, ll1_prompt, ll2_prompt],
            dim=1
        )

        space_embed = self.struct_proj(struct_feat)
        z_lor = self._to_lorentz_map(space_embed)

        new_ll3, new_lh3, new_hl3 = self.struct_decoder(z_lor)

        rec = self._iwt2_haar(new_ll3, new_lh3, new_hl3, hh3)
        rec = rec[:, :, :h, :w]

        out = self.out_proj(rec)

        return out


class UT1(nn.Module):

    def __init__(self,
                 inp_channels=3,
                 dim=48,
                 out_channels=3,
                 num_blocks=[1, 2, 3],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(UT1, self).__init__()

        self.rebnconvin = REBNCONV(inp_channels, dim, dirate=1)

        # 要是改的话，得在这里改。增加深度可分离卷积
        # 转换成目标QKV的大小

        self.relu = nn.ReLU(inplace=False)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoder_level1 = nn.Sequential(*[
            MambaBlock(dim=int(dim * 1 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                       bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.latent = REBNCONV(dim, dim, dirate=1)
        self.wlbm = WLBM(dim)

        self.decoder_level1 = nn.Sequential(*[
            MambaBlock(dim=int(dim * 1 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                       bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        # 在这变回来

        self.fuse2 = REBNCONV(dim * 2, dim, dirate=1)
        self.fuse1 = REBNCONV(dim * 2, out_channels, dirate=1)

        self.out_conv = REBNCONV(out_channels, out_channels, 1)

    def forward(self, x):
        _, _, hei, wid = x.shape

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.encoder_level1(hxin)
        hx = self.pool(hx1)

        hx = self.latent(hx)
        hx = self.wlbm(hx)

        # ------------decoder-----------

        hx1d = self.decoder_level1(hx)
        hx1dup = _upsample_like(hx1d, hx1)
        hx1f = self.fuse1(torch.cat((hx1dup, hx1), 1))

        hx_out = hx1f

        return self.out_conv(hx_out)




class MN(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, mode='train', deepsuper=True):
        super(MN, self).__init__()
        self.mode = mode
        self.deepsuper = deepsuper

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool2 = nn.MaxPool2d(4, stride=4, ceil_mode=True)
        self.pool3 = nn.MaxPool2d(8, stride=8, ceil_mode=True)
        # self.pool4 = nn.MaxPool2d(16, stride=16, ceil_mode=True)

        self.stage1 = UT1(in_ch, 32, 16)
        self.stage2 = UT1(16, 32, 32)
        self.stage3 = UT1(32, 32, 64)
        # self.stage4 = UT1(128, 48, 256)

        # self.stage4d = UT1(256, 48, 256)
        self.stage3d = UT1(64, 32, 64)
        self.stage2d = UT1(64, 32, 32)
        self.stage1d = UT1(32, 32, 16)

        # self.fuse4 = self._fuse_layer(256, 256, 256, fuse_mode='AsymBi')
        self.fuse3 = SaliencyGateSkipFusion(64, 64, 64, img_channels=in_ch)
        self.fuse2 = SaliencyGateSkipFusion(32, 32, 32, img_channels=in_ch)
        self.fuse1 = SaliencyGateSkipFusion(16, 16, 16, img_channels=in_ch) #
        self.cslr = CrossStageStructuralLorentzRefinement(c3=64, c2=32, c1=16)

        # ------------------------PDE--------------------------

        self.side1 = nn.Conv2d(64, out_ch, 1)
        self.side2 = nn.Conv2d(64, out_ch, 1)
        self.side3 = nn.Conv2d(32, out_ch, 1)
        self.side4 = nn.Conv2d(16, out_ch, 1)
        # self.side5 = nn.Conv2d(32, out_ch, 1)

        self.out_conv = nn.Conv2d(32, out_ch, 1)
        self.outconv = nn.Conv2d(4 * out_ch, out_ch, 1)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):  # fuse_mode='AsymBi'

        if fuse_mode == 'AsymBi':
            fuse_layer = Fuse(in_high_channels, in_low_channels, out_channels)
            # fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer

    def forward(self, x):
        _, _, hei, wid = x.shape
        raw = x
        hx = x

        # stage 1

        hx1 = self.stage1(hx)
        hx = self.pool(hx1)

        # stage 2

        hx2 = self.stage2(hx)
        hx = self.pool(hx2)

        # stage 3

        hx3 = self.stage3(hx)
        hx3 = self.cslr(hx3, hx2, hx1)
        hx = self.pool(hx3)


        # -------------------- decoder --------------------


        hx3d = self.stage3d(hx)  # 这里注意改
        hx3dup = _upsample_like(hx3d, hx3)
        p3 = F.adaptive_max_pool2d(raw, hx3.shape[-2:])
        hx3f = self.fuse3(hx3dup, hx3, p3)#

        hx2d = self.stage2d(hx3f)
        hx2dup = _upsample_like(hx2d, hx2)
        p2 = F.adaptive_max_pool2d(raw, hx2.shape[-2:])
        hx2f = self.fuse2(hx2dup, hx2, p2)#p2

        hx1d = self.stage1d(hx2f)
        hx1dup = _upsample_like(hx1d, hx1)
        p1 = F.adaptive_max_pool2d(raw, hx1.shape[-2:])
        hx1f = self.fuse1(hx1dup, hx1, p1)#

        # --------------------deep supervision-------------------
        if self.deepsuper:
            # d5 = F.interpolate(self.side1(hx4), size=[hei, wid])  # 这里注意改
            d4 = F.interpolate(self.side1(hx3), size=[hei, wid])
            d3 = F.interpolate(self.side2(hx3f), size=[hei, wid])
            d2 = F.interpolate(self.side3(hx2f), size=[hei, wid])
            d1 = self.side4(hx1f)
            out = self.outconv(torch.cat((d1, d2, d3, d4), 1))
            if self.mode == 'train':
                return torch.sigmoid(out)
            else:
                return torch.sigmoid(out)
        else:
            return torch.sigmoid(out=self.out_conv(hx1f))




if __name__ == '__main__':
    # 初始化模型
    model = MN(1, 1, mode='train', deepsuper=True).cuda()

    model.eval()  # 设置为评估模式

    inputs = torch.rand(1, 1, 256, 256).cuda()  # 假设输入图像大小为 256x256

    # 计算 FLOPs 和 Params
    flops, params = profile(model, (inputs,))

    #重置显存峰值统计值
    torch.cuda.reset_peak_memory_stats()

    # 计算 FPS
    num_iterations = 100  # 计算100次推理时间的平均值
    start_time = time.time()

    with torch.no_grad():  # 关闭梯度计算
        for _ in range(num_iterations):
            output = model(inputs)  # 推理过程

    torch.cuda.synchronize()
    end_time = time.time()

    # 计算每秒的图像数
    elapsed_time = end_time - start_time  # 总时间
    fps = num_iterations / elapsed_time  # 计算FPS

    #计算现存占用
    memory_usage = torch.cuda.max_memory_allocated()/(1024 ** 2)

    # 打印结果
    print("-" * 50)
    print(f'FLOPs = {flops / 1000 ** 3} G')  # FLOPs 单位为G
    print(f'Params = {params / 1000 ** 2} M')  # 参数量单位为M
    print(f'FPS = {fps:.2f}')  # 打印FPS，保留2位小数
    print(f'Memory Usage = {memory_usage:.2f}MB')