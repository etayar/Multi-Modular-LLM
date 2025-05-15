import torch
import torch.nn.functional as F
from model import GPTBackbone
from utils.tokenizer import load_tokenizer
from pathlib import Path
import argparse


def top_k_sampling(logits, top_k=10):
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    probs = F.softmax(values, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return indices.gather(-1, idx)


def generate_text(model, tokenizer, prompt, max_len=64, top_k=10, device="cpu"):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")
    input_ids = input_ids["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    generated = input_ids

    for _ in range(max_len - input_ids.shape[1]):
        logits = model(generated, attention_mask=attention_mask)
        next_token_logits = logits[:, -1, :]
        next_token = top_k_sampling(next_token_logits, top_k=top_k)

        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat([
            attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=device)
        ], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated.squeeze(), skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    ckpt = torch.load(args.ckpt, map_location=args.device)

    config = ckpt["config"]
    model = GPTBackbone(
        vocab_size=config["vocab_size"],
        max_len=config["max_len"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.1)
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    output = generate_text(model, tokenizer, args.prompt, max_len=args.max_len, top_k=args.top_k, device=args.device)
    print("\n[Generated Text]\n", output)


if __name__ == "__main__":
    main()
