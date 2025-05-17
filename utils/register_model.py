"""
This script defines a utility function for managing trained model checkpoints
intended for inference.

Purpose:
--------
During training, each run is stored in a uniquely timestamped directory under `training_runs/`.
While this is great for experiment tracking, it makes inference cumbersome because you always
need to know the exact run-time path to load the correct model.

This module solves that problem.

The `register_checkpoint()` function copies the best-performing model checkpoint from a specific
training run into a centralized and consistent directory: `inference_checkpoints/`.

Once a model is registered:
- It can be easily accessed by your inference pipeline without worrying about paths.
- You no longer need to pass the full run timestamp.
- You can maintain only the most relevant or production-ready model in one place.

Typical usage:
--------------
from utils.register_model import register_checkpoint

register_checkpoint(run_time="2025-05-17_06-51-19")

This will copy:
  training_runs/2025-05-17_06-51-19/checkpoints/GPTBackbone_best.pt
to:
  inference_checkpoints/GPTBackbone.pt

Now your `inference/generate.py` script can load the model from that single known location.
"""


from pathlib import Path
import shutil

def register_checkpoint(run_time: str, model_name: str = "GPTBackbone", base_dir: Path = None):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]

    src = base_dir / "training_runs" / run_time / "checkpoints" / f"{model_name}_best.pt"
    dst_dir = base_dir / "inference_checkpoints"
    dst_dir.mkdir(exist_ok=True)

    dst = dst_dir / f"{model_name}.pt"

    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    shutil.copy(src, dst)
    print(f"[INFO] Model registered to: {dst}")
