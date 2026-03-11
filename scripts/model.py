from PhotonNet import TemporalAttentionUNet
import torch
model = TemporalAttentionUNet(embed_dim=16, hidden_dim=32, unet_base=32).cuda()

x = torch.randn(1, 8, 800, 800, 3).cuda()  # example only
out = model(x)

print(out["recon"].shape)    # (1, 3, 800, 800)
print(out["heatmap"].shape)  # (1, 3, 800, 800)
print(out["noisy"].shape)    # (1, 3, 800, 800)
print(out["attn"].shape)     # (1, 8, 800, 800, 1)
