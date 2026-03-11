# PhotonNet: Temporal Attention U-Net for Single-Photon Reconstruction

PhotonNet reconstructs RGB images from **single-photon camera (SPC)** photoncubes (`.npy`) using **pixel-wise temporal attention** followed by a compact **U-Net**.

**Key ideas**
- Split reconstruction into temporal fusion (per-pixel attention over time) and spatial refinement (U-Net).
- Input: 8 temporal groups derived from packed SPC bitplanes.
- Outputs: reconstructed RGB image, attention heatmap, and attention weights.

**Data format**
```
data/
├── class_1/
│   ├── sample_001.npy
│   ├── sample_001.png
│   ├── sample_002.npy
│   └── sample_002.png
├── class_2/
│   └── ...
```
- Each `.npy` must have a paired `.png` with the same stem.
- The loader expects photoncubes shaped like `(1024, 800, 100, 3)` and builds 8 temporal groups.

**Dependencies**
- Python 3.9+
- PyTorch
- numpy, pillow, tqdm, torchmetrics

**Train**
```
python scripts/cli.py \
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
```
- Train/val split is 80/20.

**Outputs**
- `runs_spc/logs/metrics.csv`
- `runs_spc/checkpoints/latest.pt`
- `runs_spc/checkpoints/best_val_loss.pt`
- `runs_spc/checkpoints/best_lpips.pt`

**Inference**
```
python scripts/Test.py \
  --test_dir ./data/test \
  --ckpt ./runs_spc/checkpoints/best_lpips.pt \
  --output_dir ./inference \
  --device cuda:0
```
- Outputs reconstructed `.png` files under `inference/<class_name>/`.

**Notes**
- The code uses AMP on CUDA but will fall back to CPU if CUDA is unavailable.

**License**
Apache License 2.0. See `LICENSE` and `NOTICE`.
