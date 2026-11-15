# Install

Python 3.10+ (developed on 3.11). Install PyTorch for your hardware FIRST (pip's default torch is CPU-only on Windows and may not match your CUDA on Linux), then the rest.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows git-bash: source .venv/Scripts/activate
# GPU (CUDA 12.8 wheels, Linux + Windows):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CPU only (fine for eval and video, slow for training):
# pip install torch torchvision
pip install -r requirements.txt
python -m pytest tests -q          # ~40 s, 14 tests
```

Rendering is offscreen (MuJoCo's default GL backend). On a headless Linux box set `MUJOCO_GL=egl` (or `osmesa`).

The `AdroitHand... reward functions were updated` warning on every import comes from gymnasium-robotics and is unrelated to this project.

Hardware used for every number in RESULTS.md: HP OMEN laptop, RTX 4060 Laptop 8 GB, i7-14700HX, 16 GB RAM, Windows 11.
