import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet3DSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=8):
        super().__init__()

        self.enc1 = DoubleConv3D(in_ch, base)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.enc2 = DoubleConv3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.enc3 = DoubleConv3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.bottleneck = DoubleConv3D(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = DoubleConv3D(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = DoubleConv3D(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = DoubleConv3D(base * 2, base)

        self.out = nn.Conv3d(base, out_ch, kernel_size=1)

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


class TemporalAttentionUNet(nn.Module):
    def __init__(self, embed_dim=16, hidden_dim=32, unet_base=8):
        super().__init__()
        self.unet3d = UNet3DSmall(in_ch=3, out_ch=3, base=unet_base)

    def forward(self, x):
        # x: (B, T, H, W, C)
        x_3d = x.permute(0, 4, 1, 2, 3).contiguous()   # (B, C, T, H, W)

        feat_3d = self.unet3d(x_3d)                    # (B, 3, T, H, W)

        heatmap = feat_3d.mean(dim=2)                  # (B, 3, H, W)
        recon = heatmap                                # keep interface compatible
        noisy = x.sum(dim=1).permute(0, 3, 1, 2).contiguous()

        B, T, H, W, C = x.shape
        attn = torch.full(
            (B, T, H, W, 1),
            fill_value=1.0 / T,
            device=x.device,
            dtype=x.dtype
        )

        return {
            "recon": recon,
            "heatmap": heatmap,
            "noisy": noisy,
            "attn": attn,
        }
