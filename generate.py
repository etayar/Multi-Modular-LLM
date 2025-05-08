import torch
from model import GPTBackbone
from utils.tokenizer import load_tokenizer
from pathlib import Path
import argparse


def load_model(config, checkpoint_path):
    model = GPTBackbone(
        vocab_size=config["vocab_size"],
        max_len=config["max_len"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"]
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate_text(prompt, model, tokenizer, config, max_length=50):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_ids = input_ids[:, :config["max_len"]]
    generated = input_ids.clone()

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

        if generated.size(1) >= config["max_len"]:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    args = parser.parse_args()

    # must match training config
    config = {
        "vocab_size": 1000,
        "max_len": 64,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2
    }

    tokenizer = load_tokenizer()
    model = load_model(config, Path(args.checkpoint))
    output = generate_text(args.prompt, model, tokenizer, config, args.max_length)
    print("\n📝 Generated text:")
    print(output)


if __name__ == "__main__":
    main()
