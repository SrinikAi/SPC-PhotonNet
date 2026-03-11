import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from PhotonNet import TemporalAttentionUNet


def make_8_groups(npy_path):
    photoncube = np.load(npy_path, mmap_mode="r")  # (1024, 800, 100, 3)
    parts = np.zeros((8, 800, 800, 3), dtype=np.uint16)

    frames_per_group = photoncube.shape[0] // 8
    for i in range(8):
        chunk = photoncube[i * frames_per_group:(i + 1) * frames_per_group]
        chunk_bits = np.unpackbits(chunk, axis=2)   # -> (128, 800, 800, 3)
        parts[i] = chunk_bits.sum(axis=0, dtype=np.uint16)

    parts = parts.astype(np.float32)
    parts /= (parts.max() + 1e-8)
    parts = torch.from_numpy(parts)  # (8, 800, 800, 3)
    return parts


def load_model(ckpt_path, device, embed_dim=16, hidden_dim=32, unet_base=32):
    model = TemporalAttentionUNet(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        unet_base=unet_base
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def save_tensor_as_png(tensor, save_path):
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = (tensor * 255.0).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


@torch.no_grad()
def run_inference(model, test_dir, output_dir, device):
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)

    npy_files = []
    for class_dir in sorted(test_dir.iterdir()):
        if class_dir.is_dir():
            npy_files.extend(sorted(class_dir.glob("*.npy")))

    if len(npy_files) == 0:
        raise RuntimeError(f"No .npy files found under {test_dir}")

    for npy_path in tqdm(npy_files, desc="Inference"):
        class_name = npy_path.parent.name
        out_path = output_dir / class_name / f"{npy_path.stem}.png"

        x = make_8_groups(npy_path).unsqueeze(0).to(device)  # (1, 8, 800, 800, 3)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
            recon = out["recon"][0]  # (3, 800, 800)

        save_tensor_as_png(recon, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test directory with class subfolders")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="inference", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--unet_base", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(
        ckpt_path=args.ckpt,
        device=device,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        unet_base=args.unet_base,
    )

    run_inference(
        model=model,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
