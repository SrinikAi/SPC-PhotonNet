import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


class PixelTemporalAttention(nn.Module):
    def __init__(self, in_ch=3, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, embed_dim)
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.out_proj = nn.Linear(embed_dim, in_ch)

    def forward(self, x):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape

        x_seq = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)

        z = self.in_proj(x_seq)
        scores = self.score(z)
        weights = torch.softmax(scores, dim=1)
        fused = (z * weights).sum(dim=1)
        out = self.out_proj(fused)

        heatmap = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        attn = weights.view(B, H, W, T, 1).permute(0, 3, 1, 2, 4).contiguous()
        return heatmap, attn


class TemporalAttentionUNet(nn.Module):
    def __init__(self, embed_dim=16, hidden_dim=32, unet_base=32):
        super().__init__()
        self.temporal_attention = PixelTemporalAttention(
            in_ch=3,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )
        self.unet = UNetSmall(in_ch=3, out_ch=3, base=unet_base)

    def forward(self, x):
        heatmap, attn = self.temporal_attention(x)   # (B, 3, H, W)
        out = self.unet(heatmap)                     # (B, 3, H, W)

        noisy = x.sum(dim=1).permute(0, 3, 1, 2).contiguous()

        return {
            "recon": out,
            "heatmap": heatmap,
            "noisy": noisy,
            "attn": attn
        }
