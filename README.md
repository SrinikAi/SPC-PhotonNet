# SPC-PhotonNet
SPC-PhotonNet is a PyTorch implementation for Single-photon camera reconstruction with pixel-wise temporal attention and U-Net refinement.

# PhotonNet: Temporal Attention U-Net for Single-Photon Camera Reconstruction

PhotonNet is a PyTorch-based reconstruction model for **single-photon camera (SPC)** data. It converts photon cube measurements stored as `.npy` files into RGB image reconstructions by combining a pixel-wise temporal attention module with a compact U-Net decoder. [file:1][file:2]

The model is designed for paired training with `.npy` photon measurements and corresponding `.png` RGB targets. The preprocessing pipeline aggregates photon events into 8 temporal groups, learns how much each temporal group should contribute at each pixel, and then refines the fused representation with a convolutional encoder-decoder network. [file:1][file:2]

## Overview

Single-photon camera data is highly sparse and noisy, so directly reconstructing a clean image from raw measurements is difficult. In this implementation, the input photon cube is first transformed into a compact set of temporal summaries, then the model learns a per-pixel weighting over those summaries before image reconstruction. [file:1][file:2]

The overall architecture has two main stages:  
1. **PixelTemporalAttention** for temporal fusion across grouped photon measurements. [file:2]  
2. **UNetSmall** for spatial refinement and final RGB reconstruction. [file:2]

This gives the model a clear division of labor: the attention block decides **when** information is useful, while the U-Net decides **how** to spatially reconstruct a clean image from the fused response. [file:2]

## Input representation

The training dataset expects class-wise folders containing paired `.npy` and `.png` files with the same base filename. The loader searches each class directory, pairs every `.npy` file with a `.png` file of the same stem, and uses those pairs for supervised training. [file:1]

Each `.npy` file is treated as a photon cube with shape `(1024, 800, 100, 3)`. The preprocessing routine divides the 1024 frames into 8 equal temporal groups, so each group contains 128 frames. [file:1]

For each temporal group:
- A slice of shape `(128, 800, 100, 3)` is extracted. [file:1]
- `np.unpackbits(..., axis=2)` expands the packed bit dimension from 100 to 800, producing `(128, 800, 800, 3)`. [file:1]
- The 128 frames are summed along the temporal axis, giving a photon accumulation image of shape `(800, 800, 3)` for that group. [file:1]

After all 8 groups are processed, the result is a tensor of shape `(8, 800, 800, 3)`. This tensor is normalized by its global maximum value and converted to a PyTorch tensor before being passed to the model. [file:1]

So the network input for one sample is:

```text
x: (8, 800, 800, 3)
