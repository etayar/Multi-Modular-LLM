# 🧠 Multi-Modular LLM — Transformer Training Script

This project implements a clean, modular pipeline to train a GPT-style transformer model using PyTorch. It includes causal masking, real tokenization (via Hugging Face), logging, checkpointing, and future support for crawling text from the web.

---

## 📁 Project Structure

.
├── data/
│   └── train.txt                  # Text training data (one sentence per line)
├── logs/
│   └── train_log.csv              # Epoch metrics with config and perplexity
├── checkpoints/
│   └── checkpoint_epoch_*.pt     # Saved model weights
├── model/
│   └── transformer.py            # GPT backbone with causal masking
├── utils/
│   └── tokenizer.py              # Loads Hugging Face tokenizer
├── scripts/
│   └── generate_text_dataset.py  # Generates local dataset for testing
├── training/
│   └── train_transformer.py      # Full training loop
└── README.md                     # You are here

---

## 🧪 Training Workflow

1. Generate data:
   python scripts/generate_text_dataset.py

2. Train the model:
   python training/train_transformer.py

3. Check logs and checkpoints:
   - logs/train_log.csv
   - checkpoints/checkpoint_epoch_*.pt

---

## ⚙️ Configuration

Defined in `get_config()` in `train_transformer.py`:

- vocab_size, max_len, embed_dim, num_heads, num_layers
- batch_size, lr, epochs
- dataset_source: \"local\" or \"web\" (placeholder for now)


