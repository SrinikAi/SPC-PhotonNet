import numpy as np
import matplotlib.pyplot as plt

npy_path = "/home/skachavarapu/home/spc/train/classroom/000000.npy"

# Load lazily from disk
photoncube = np.load(npy_path, mmap_mode="r")
print("photoncube shape:", photoncube.shape)
print("dtype:", photoncube.dtype)

# Example: first 10 temporal slices
subset = photoncube[:10]
print("subset shape:", subset.shape)

# Unpack packed bits along axis=2
bitplanes = np.unpackbits(subset, axis=2)
print("bitplanes shape:", bitplanes.shape)

# Pick one sample to visualize
t = 0          # first frame from the 10 selected
c = 0          # first channel
bit_idx = 0    # first unpacked bit-plane

# 2D image for one bit-plane and one channel
img = bitplanes[t, :, :, c]

plt.figure(figsize=(8, 8))
plt.imshow(img, cmap="gray", interpolation="nearest")
plt.title(f"Bitplane view | frame={t}, channel={c}")
plt.axis("off")
plt.show()

# Visualize multiple channels for the same frame
fig, axes = plt.subplots(1, min(3, bitplanes.shape[-1]), figsize=(15, 5))
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

for ch in range(min(3, bitplanes.shape[-1])):
    axes[ch].imshow(bitplanes[t, :, :, ch], cmap="gray", interpolation="nearest")
    axes[ch].set_title(f"Frame {t}, Channel {ch}")
    axes[ch].axis("off")

plt.tight_layout()
plt.show()

# Optional: sum across channels to see activity map
activity = bitplanes[t].sum(axis=-1)

plt.figure(figsize=(8, 8))
plt.imshow(activity, cmap="hot", interpolation="nearest")
plt.title(f"Summed activity map | frame={t}")
plt.colorbar()
plt.axis("off")
plt.show()
