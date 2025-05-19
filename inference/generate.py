import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from model import GPTBackbone
from utils.tokenizer import load_tokenizer
import argparse
import json
from torch.cuda.amp import autocast


def top_k_sampling(logits, top_k=10):
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    probs = F.softmax(values, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return indices.gather(-1, idx)


def top_p_sampling(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = probs.cumsum(dim=-1)

    keep_mask = cumulative_probs <= top_p
    keep_mask[..., 1:] = keep_mask[..., :-1]
    keep_mask[..., 0] = 1

    filtered_logits = sorted_logits.masked_fill(~keep_mask, float('-inf'))
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    idx = torch.multinomial(filtered_probs, num_samples=1)
    return sorted_indices.gather(-1, idx)


def greedy_sampling(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)


def generate_text(
    model, tokenizer, prompt, config,
    sampling="top_k", max_len=64, top_k=10, top_p=0.9,
    temperature=1.0, device="cpu"
):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    generated = input_ids

    max_gen_len = min(config.get("max_len", 1024), max_len)

    for _ in range(max_gen_len - input_ids.shape[1]):
        with torch.no_grad(), autocast():
            logits = model(generated, attention_mask=attention_mask)
            next_token_logits = logits[:, -1, :] / temperature

            if sampling == "greedy":
                next_token = greedy_sampling(next_token_logits)
            elif sampling == "top_p":
                next_token = top_p_sampling(next_token_logits, top_p)
            else:
                next_token = top_k_sampling(next_token_logits, top_k)

        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=device)],
            dim=1
        )

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated.squeeze(), skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--sampling", type=str, default="top_k", choices=["greedy", "top_k", "top_p"])
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_to", type=str, default=None)

    # Optional config if missing in ckpt
    parser.add_argument("--config_json", type=str, default=None, help="Path to JSON file with model config")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    ckpt = torch.load(args.ckpt, map_location=args.device)

    if "config" in ckpt:
        config = ckpt["config"]
    elif args.config_json:
        print(f"[INFO] 'config' not found in checkpoint — loading from {args.config_json}")
        with open(args.config_json, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(
            "'config' not found in checkpoint and --config_json not provided.\n"
            "You must pass a config JSON file containing model params."
        )

    model = GPTBackbone(
        vocab_size=config["vocab_size"],
        max_len=config["max_len"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.1)
    ).to(args.device)

    model.load_state_dict(ckpt["model_state_dict"])

    print("[INFO] Generating text...")
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        config=config,
        sampling=args.sampling,
        max_len=args.max_len,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        device=args.device
    )

    print("\n[Generated Text]")
    print(output)

    if args.save_to:
        with open(args.save_to, "w") as f:
            f.write(output)
        print(f"[INFO] Output saved to {args.save_to}")


if __name__ == "__main__":
    main()
