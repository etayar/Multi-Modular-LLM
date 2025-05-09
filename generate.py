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


def generate_text(prompt, model, tokenizer, config, max_length=50, temperature=1.0, top_k=0, top_p=0.0):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_ids = input_ids[:, :config["max_len"]]
    generated = input_ids.clone()

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k)
                probs = torch.softmax(values, dim=-1)
                next_token = indices.gather(-1, torch.multinomial(probs, num_samples=1))
            elif top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_keep = cumulative_probs <= top_p
                sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
                sorted_indices_to_keep[..., 0] = 1
                filtered_logits = sorted_logits.masked_fill(~sorted_indices_to_keep, float('-inf'))
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = sorted_indices.gather(-1, torch.multinomial(probs, num_samples=1))
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

        generated = torch.cat((generated, next_token), dim=1)

        if generated.size(1) >= config["max_len"]:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
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
    output = generate_text(
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        config=config,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    print("\n📝 Generated text:")
    print(output)


if __name__ == "__main__":
    main()
