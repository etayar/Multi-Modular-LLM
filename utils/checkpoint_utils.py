from typing import List
from pathlib import Path

def list_checkpoints(model_name: str, dataset_name: str, ckpt_dir: str) -> List[Path]:
    """
    Return all checkpoint files sorted by modification time (oldest to newest).
    Compatible with any environment, including Colab and notebooks.
    """
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    ckpt_path = base_dir / ckpt_dir
    pattern = f"{model_name}_{dataset_name}_epoch_*.pt"
    return sorted(ckpt_path.glob(pattern), key=lambda p: p.stat().st_mtime)

# Example usage:
if __name__ == "__main__":
    model = "GPTBackbone"
    dataset = "wikipedia"
    ckpt_dir = "checkpoints"

    print(f"Looking for checkpoints in: {ckpt_dir}")
    ckpts = list_checkpoints(model, dataset, ckpt_dir)

    if not ckpts:
        print("⚠️ No checkpoints found.")
    else:
        print("\nAvailable checkpoints:")
        for ckpt in ckpts:
            print(" -", ckpt.name)

