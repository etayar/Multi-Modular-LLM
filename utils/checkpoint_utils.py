from typing import List
from pathlib import Path

def list_all_checkpoints(ckpt_dir: str) -> List[Path]:
    """
    Return all .pt checkpoint files sorted by modification time.
    Works in any environment (Colab, local, server).
    """
    try:
        base_dir = Path(__file__).resolve().parent.parent
    except NameError:
        base_dir = Path.cwd()

    # Allow absolute paths to override base_dir
    ckpt_path = Path(ckpt_dir) if Path(ckpt_dir).is_absolute() else base_dir / ckpt_dir
    return sorted(ckpt_path.glob("*.pt"), key=lambda p: p.stat().st_mtime)

# Example usage
if __name__ == "__main__":
    ckpt_dir = "checkpoints"  # just the name of the folder relative to project root
    print(f"Looking for checkpoints in: {ckpt_dir}")
    ckpts = list_all_checkpoints(ckpt_dir)

    if not ckpts:
        print("⚠️ No checkpoints found.")
    else:
        print("\nAvailable checkpoints:")
        for ckpt in ckpts:
            print(" -", ckpt.name)
