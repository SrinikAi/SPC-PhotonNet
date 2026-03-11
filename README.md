# PhotonNet: Temporal Attention U-Net for Single-Photon Camera Reconstruction

PhotonNet is a PyTorch implementation for reconstructing RGB images from **single-photon camera (SPC)** measurements stored as `.npy` photon cubes. The model combines **pixel-wise temporal attention** with a compact **U-Net** to fuse sparse photon observations and generate clean image reconstructions.

## Overview

Single-photon camera data is sparse, noisy, and temporally distributed. PhotonNet addresses this by splitting the reconstruction problem into two stages:

1. **Temporal fusion** across grouped photon measurements.
2. **Spatial reconstruction** using a U-Net decoder.

This design lets the model first decide which temporal groups are most informative at each pixel, then refine the fused response into a final RGB image.


### Data set format
data/
├── class_1/
│   ├── sample_001.npy
│   ├── sample_001.png
│   ├── sample_002.npy
│   └── sample_002.png
├── class_2/
│   ├── sample_101.npy
│   ├── sample_101.png
│   └── ...

### Training command
python cli.py \
  --data_path ./data/train \
  --batch_size 1 \
  --lr 2e-4 \
  --device cuda:0 \
  --epochs 100 \
  --save_dir ./runs_spc \
  --num_workers 10 \
  --embed_dim 16 \
  --hidden_dim 32 \
  --unet_base 32 \
  --heatmap_loss_weight 0.3
  
Repository structure

spc-photonnet/
├── PhotonNet.py
├── cli.py
├── inference.py
├── README.md
└── requirements.txt
