from pathlib import Path
from typing import List

def list_all_checkpoints(ckpt_dir: str) -> List[Path]:
    """
    Return all .pt checkpoint files sorted by modification time.
    """
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    ckpt_path = base_dir / ckpt_dir if not ckpt_dir.startswith("/") else Path(ckpt_dir)
    return sorted(ckpt_path.glob("*.pt"), key=lambda p: p.stat().st_mtime)

# Example usage
if __name__ == "__main__":
    ckpt_dir = "/content/drive/MyDrive/llm_checkpoints"
    print(f"Looking for checkpoints in: {ckpt_dir}")
    ckpts = list_all_checkpoints(ckpt_dir)

    if not ckpts:
        print("⚠️ No checkpoints found.")
    else:
        print("\nAvailable checkpoints:")
        for ckpt in ckpts:
            print(" -", ckpt.name)
