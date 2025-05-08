from pathlib import Path

def generate_dataset(filepath: Path, repeat: int = 100):
    lines = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformers are powerful models for sequence learning.",
        "PyTorch is a popular deep learning framework.",
        "Large language models are trained on vast text corpora.",
        "Causal masking prevents tokens from seeing the future.",
        "Positional embeddings encode word order information.",
        "Attention is all you need.",
        "GPT models use decoder-only transformers.",
        "Training data diversity improves generalization.",
        "Learning rate schedules help stabilize training."
    ]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        for line in lines * repeat:
            f.write(line + "\n")
    print(f"Text dataset created at: {filepath}")

def main():
    dataset_path = Path("data/train.txt")
    generate_dataset(dataset_path)

if __name__ == "__main__":
    # python scripts/generate_text_dataset.py
    main()
