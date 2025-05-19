import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import GPTBackbone
from model.utils import count_parameters
from utils.tokenizer import load_tokenizer
from torch.utils.data import DataLoader, IterableDataset
from pathlib import Path
import json
import subprocess
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """Patience-based early stopping with configurable Δ-threshold."""

    def __init__(self, patience: int = 5, es_threshold: float = 1e-3):
        self.patience = patience
        self.es_threshold = es_threshold
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.es_threshold:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
        else:
            self.counter += 1
        return self.counter >= self.patience


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class TransformerTrainer:
    """Training loop for GPTBackbone with streaming / local datasets."""

    def __init__(self, config: dict):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.config = config
        self.config["__data_mode__"] = "streaming" if config.get("use_streaming") else "local"
        self.config["__git_commit__"] = self._git_commit()
        self.config["__run_time__"] = config.get("resume_from_date") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.device = config["device"]

        self.tokenizer = load_tokenizer()
        self._ensure_vocab_size()
        self.model = self._build_model()

        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id for ignore_index loss.")
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.endswith("bias") or any(k in name for k in ["LayerNorm", "layer_norm", ".ln_"]):
                    no_decay.append(param)
                else:
                    decay.append(param)

        optimizer_grouped = [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        self.optimizer = optim.AdamW(optimizer_grouped, lr=config["lr"])
        self.scaler = GradScaler()

        try:
            self.project_root = Path(__file__).resolve().parents[1]
        except NameError:
            self.project_root = Path.cwd()

        self.run_dir = self.project_root / "training_runs" / self.config["__run_time__"]

        # ✅ Custom checkpoint directory logic
        if config.get("ckpt_dir"):
            self.ckpt_dir = Path(config["ckpt_dir"])
        else:
            self.ckpt_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Saving logs to {self.log_dir}")

        self.lr_history, self.train_loss_history, self.eval_loss_history = [], [], []
        self.best_eval_loss = float("inf")
        self.early_stopping = EarlyStopping(patience=config.get("early_stop_patience", 3))
        self.start_epoch = 1

        self.dataset = self._build_dataset()
        self.dataloader = self._build_dataloader()
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
        (self.run_dir / "config_snapshot.json").write_text(json.dumps(self.config, indent=2))

        if config.get("resume_from_date"):
            self._load_checkpoint(config["resume_from_date"])

    def _git_commit(self):
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except Exception:
            return "N/A"

    def _ensure_vocab_size(self):
        vocab = self.tokenizer.vocab_size
        if self.config.get("vocab_size") is None or self.config["vocab_size"] < vocab:
            print(f"[WARN] vocab_size updated to tokenizer size ({vocab})")
            self.config["vocab_size"] = vocab

    def _build_model(self):
        m = GPTBackbone(
            vocab_size=self.config["vocab_size"],
            max_len=self.config["max_len"],
            embed_dim=self.config["embed_dim"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            dropout=self.config.get("dropout", 0.15),
        )
        print(f"Model initialized with {count_parameters(m):,} parameters.")
        return m.to(self.device)

    def _build_dataset(self):
        cfg = self.config
        if cfg.get("use_wikipedia"):
            print("[INFO] Using Hugging Face Wikipedia stream dataset")
            from datasets import load_dataset

            class WikipediaStream(IterableDataset):
                def __init__(self, tokenizer, chunk_len, max_articles=None):
                    self.tokenizer = tokenizer
                    self.chunk_len = chunk_len
                    self.max_articles = max_articles
                    self.dataset = load_dataset("wikipedia", cfg.get("dataset_config", "20220301.en"), split="train", streaming=True)

                def __iter__(self):
                    seen = 0
                    for ex in self.dataset:
                        if self.max_articles and seen >= self.max_articles:
                            break
                        tokens = self.tokenizer(ex["text"], truncation=True, max_length=min(self.chunk_len * 10, 2048), return_attention_mask=False)["input_ids"]
                        for i in range(0, len(tokens) - self.chunk_len + 1, self.chunk_len):
                            chunk = tokens[i: i + self.chunk_len]
                            yield {
                                "input_ids": torch.tensor(chunk),
                                "attention_mask": torch.ones(len(chunk), dtype=torch.long),
                            }
                        seen += 1

            return WikipediaStream(self.tokenizer, cfg["max_len"], cfg.get("max_articles"))

        if cfg.get("use_streaming"):
            print("[INFO] Using WebCrawlStreamDataset (live crawling)")
            from utils.web_streaming_dataset import WebCrawlStreamDataset
            return WebCrawlStreamDataset(
                urls=cfg["webcrawl_urls"],
                tokenizer=self.tokenizer,
                max_length=cfg["max_len"],
                delay=cfg.get("crawl_delay", 1.0),
            )

        print(f"[INFO] Using local text dataset: {self.project_root / cfg['dataset_path']}")
        from utils.dataset import TextDataset
        return TextDataset(
            str(self.project_root / cfg["dataset_path"]),
            self.tokenizer,
            max_length=cfg["max_len"],
        )

    def _build_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=not self.config.get("use_streaming"),
            num_workers=(os.cpu_count() or 2) if not self.config.get("use_streaming") else 0,
        )

    def _load_checkpoint(self, date_str):
        ckpt = (
            self.project_root
            / "training_runs"
            / date_str
            / self.config["ckpt_dir"]
            / f"{self.config['__model_name__']}_best.pt"
        )
        if not ckpt.exists():
            print(f"[WARN] Checkpoint not found: {ckpt}")
            return
        print(f"[INFO] Loading checkpoint from {ckpt.name}")
        data = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.start_epoch = data["epoch"] + 1
        self.lr_history = data.get("lr_history", [])
        self.train_loss_history = data.get("train_loss_history", [])
        self.eval_loss_history = data.get("eval_loss_history", [])
        self.config.update(data.get("config", {}))
        if "early_stop_state" in data:
            self.early_stopping.__dict__.update(data["early_stop_state"])
        print(f"[INFO] Resuming from epoch {self.start_epoch}")

    @staticmethod
    def _shift(inputs, attn):
        dec_inputs = inputs[:, :-1].contiguous()
        dec_attn = attn[:, :-1].contiguous()
        labels = inputs[:, 1:].contiguous()
        return dec_inputs, dec_attn, labels

    def train_step(self, batch):
        inputs = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        dec_inputs, dec_attn, labels = self._shift(inputs, attn)

        self.model.train()
        with autocast():
            logits = self.model(dec_inputs, attention_mask=dec_attn)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.tb_writer.add_scalar("GradNorm", grad_norm.item())

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.tb_writer.add_scalar("Loss/batch_train", loss.item())
        return loss.item()

    @torch.no_grad()
    def evaluate(self, num_batches=None):
        self.model.eval()
        total = 0.0
        count = 0
        num_batches = num_batches or self.config.get("max_eval_batches") or 5
        for i, batch in tqdm(enumerate(self.dataloader), total=num_batches, desc="Evaluating", leave=False):
            if i >= num_batches:
                break
            try:
                inputs = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                dec_inputs, dec_attn, labels = self._shift(inputs, attn)
                logits = self.model(dec_inputs, attention_mask=dec_attn)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total += loss.item()
                count += 1
            except Exception as e:
                print(f"[WARN] Skipping eval batch {i}: {e}")

        if count == 0:
            print("[WARN] No valid eval batches — skipping validation")
            return float("inf"), float("inf")

        avg_loss = total / count
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Validation — Avg Loss {avg_loss:.4f} | Perplexity {ppl:.2f}")
        return avg_loss, ppl

    def train(self):
        print(f"\nRUN TIME: {self.config['__run_time__']}\n")
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            self.dataloader = self._build_dataloader()
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            train_losses = []
            max_batches = self.config.get("max_train_batches")
            pbar = tqdm(enumerate(self.dataloader), total=max_batches, desc="Training", leave=False)

            for i, batch in pbar:
                if max_batches and i >= max_batches:
                    break
                try:
                    loss = self.train_step(batch)
                    train_losses.append(loss)
                    pbar.set_postfix(loss=f"{sum(train_losses)/len(train_losses):.4f}")
                except Exception as e:
                    print(f"[WARN] Skipping train batch {i}: {e}")

            avg_train = sum(train_losses) / len(train_losses)
            eval_loss, ppl = self.evaluate()

            self.train_loss_history.append(avg_train)
            self.eval_loss_history.append(eval_loss)
            self.lr_history.append(self.scheduler.get_last_lr()[0])

            self.tb_writer.add_scalar("Loss/epoch_train", avg_train, epoch)
            self.tb_writer.add_scalar("Loss/epoch_eval", eval_loss, epoch)
            self.tb_writer.add_scalar("Perplexity/epoch_eval", ppl, epoch)
            self.tb_writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            print(f"Epoch {epoch} Complete — Train {avg_train:.4f} | Eval {eval_loss:.4f} | PPL {ppl:.2f}")

            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                ckpt = self.ckpt_dir / f"{self.config['__model_name__']}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss_history": self.train_loss_history,
                    "eval_loss_history": self.eval_loss_history,
                    "lr_history": self.lr_history,
                    "early_stop_state": self.early_stopping.__dict__,
                    "config": self.config
                }, ckpt)
                print(f"[INFO] Best model saved to {ckpt}")

            self.scheduler.step()

            if self.early_stopping.step(eval_loss):
                print(f"Early stopping triggered at epoch {epoch}.")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss_history": self.train_loss_history,
                    "eval_loss_history": self.eval_loss_history,
                    "lr_history": self.lr_history,
                    "early_stop_state": self.early_stopping.__dict__,
                    "config": self.config
                }, self.ckpt_dir / f"{self.config['__model_name__']}_earlystop.pt")
                break

        (self.log_dir / "loss_history.json").write_text(json.dumps({
            "train_loss": self.train_loss_history,
            "eval_loss": self.eval_loss_history,
            "lr": self.lr_history,
        }))
        with open(self.log_dir / "loss_history.csv", "w") as f:
            f.write("epoch,train_loss,eval_loss,lr\n")
            for ep, (tr, ev, lr) in enumerate(zip(self.train_loss_history, self.eval_loss_history, self.lr_history), start=self.start_epoch):
                f.write(f"{ep},{tr},{ev},{lr}\n")
        print(f"[INFO] Loss history saved to {self.log_dir}")


def get_config(preset="base"):
    presets = {
        "micro":  {"embed_dim": 64,  "num_heads": 2,  "num_layers": 1},
        "tiny":   {"embed_dim": 128, "num_heads": 4,  "num_layers": 2},
        "small":  {"embed_dim": 192, "num_heads": 6,  "num_layers": 4},
        "base":   {"embed_dim": 256, "num_heads": 8,  "num_layers": 6},
        "medium": {"embed_dim": 512, "num_heads": 8,  "num_layers": 8},
        "large":  {"embed_dim": 512, "num_heads": 8,  "num_layers": 12},
        "xlarge": {"embed_dim": 768, "num_heads": 12, "num_layers": 24},
    }
    assert preset in presets, f"Invalid preset '{preset}'"

    return {
        "vocab_size": None,
        "max_len": 256,
        "batch_size": 32,
        "dropout": 0.2,
        "lr": 1e-4,
        "max_articles": 5e4,
        "epochs": 15,
        "dataset_path": "data/train.txt",
        "dataset_name": "wikipedia",
        "use_wikipedia": True,
        "dataset_config": "20220301.en",
        "split": "train",
        "use_streaming": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_train_batches": None,
        "max_eval_batches": None,
        "lr_step_size": 2,
        "lr_gamma": 0.5,
        "resume_from_date": None,
        "__model_name__": "GPTBackbone",
        "crawl_delay": 1.0,
        **presets[preset],
        "webcrawl_urls": [
            "https://www.bbc.com/news/world-europe-68867750",
            "https://www.theguardian.com/world/2024/may/01/ukraine-war-frontline-report",
            "https://www.aljazeera.com/news/2024/5/14/gaza-ceasefire-latest",
            "https://arxiv.org/abs/2405.01700",
            "https://arxiv.org/abs/2405.00566",
            "https://www.gutenberg.org/files/11/11-h/11-h.htm",
            "https://www.gutenberg.org/files/98/98-h/98-h.htm",
        ],
    }


if __name__ == "__main__":
    print("[WARN] Do not run this script directly; import it or use an external launcher.")
    import sys
    sys.exit(1)
