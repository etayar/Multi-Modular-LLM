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


class EarlyStopping:
    def __init__(self, patience=5, es_threshold=1e-3):
        """
        Early stops training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How many epochs to wait before stopping if no improvement.
            es_threshold (float): Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.es_threshold = es_threshold
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        """
        Call this function after each epoch to check if training should stop.
        Returns True if training should stop, False otherwise.
        """

        if val_loss < self.best_loss - self.es_threshold:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if loss improves
        elif val_loss < self.best_loss:
            self.best_loss = val_loss  # Update best loss but don't reset patience if improvement is small
        else:
            self.counter += 1  # Increase counter if no improvement

        return self.counter >= self.patience  # Stop if patience threshold is reached


class TransformerTrainer:
    def __init__(self, config):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.config = config
        self.config["__data_mode__"] = "streaming" if config.get("use_streaming", False) else "local"
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except Exception:
            commit_hash = "N/A"
        self.config["__git_commit__"] = commit_hash
        self.config["__run_time__"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.device = config["device"]
        self.tokenizer = load_tokenizer()
        true_vocab_size = self.tokenizer.vocab_size
        if self.config.get("vocab_size") is None or self.config["vocab_size"] < true_vocab_size:
            print(f"[WARN] vocab_size was {self.config.get('vocab_size')}, updating to match tokenizer ({true_vocab_size})")
            self.config["vocab_size"] = true_vocab_size

        self.model = self._build_model()
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id set, which is required for ignore_index.")
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.scaler = GradScaler()

        try:
            self.project_root = Path(__file__).resolve().parents[1]
        except NameError:
            self.project_root = Path.cwd()

        self.lr_history = []
        self.train_loss_history = []
        self.eval_loss_history = []

        self.early_stop_patience = config.get("early_stop_patience", 3)  # Configure it in get_config()
        self.early_stopping = EarlyStopping(patience=self.early_stop_patience)

        self.start_epoch = 1
        resume_from_date = config.get("resume_from_date")
        if resume_from_date:
            self.config["__run_time__"] = resume_from_date
            ckpt_path = (
                    self.project_root
                    / "training_runs"
                    / resume_from_date
                    / config["ckpt_dir"]
                    / f"{config['__model_name__']}_best.pt"
            )
            if ckpt_path.exists():
                print(f"[INFO] Loading checkpoint from: {ckpt_path.name}")
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "early_stop_state" in checkpoint:
                    self.early_stopping = EarlyStopping()  # re-init instance
                    self.early_stopping.__dict__.update(checkpoint["early_stop_state"])
                self.start_epoch = checkpoint["epoch"] + 1
                self.lr_history = checkpoint.get("lr_history", [])
                self.train_loss_history = checkpoint.get("train_loss_history", [])
                self.eval_loss_history = checkpoint.get("eval_loss_history", [])
                print(f"[INFO] Resuming from epoch {self.start_epoch}")
            else:
                print(f"[WARN] Specified checkpoint not found: {ckpt_path}")

        self.run_output_dir = self.project_root / "training_runs" / self.config["__run_time__"]
        self.ckpt_dir = self.run_output_dir / "checkpoints"
        self.log_dir = self.run_output_dir / "logs"
        self.data_path = self.project_root / config["dataset_path"]

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Saving logs to: {self.log_dir}")
        self.log_path = self.log_dir / "train_log.csv"

        config_path = self.run_output_dir / "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        self.best_eval_loss = float("inf")

        if config.get("use_wikipedia", False):
            from datasets import load_dataset
            print(f"[INFO] Using Hugging Face Wikipedia stream dataset")

            class WikipediaStreamDataset(IterableDataset):
                def __init__(self, tokenizer, max_length, max_articles=None):
                    self.tokenizer = tokenizer
                    self.chunk_length = max_length
                    self.tokenizer_max_length = min(max_length * 10, 2048)  # allow large context, capped at 2048
                    self.max_articles = max_articles
                    self.dataset = load_dataset(
                        "wikipedia", config.get("dataset_config", "20220301.en"),
                        split="train", streaming=True
                    )

                def __iter__(self):
                    count = 0
                    for example in self.dataset:
                        if self.max_articles is not None and count >= self.max_articles:
                            break
                        tokens = self.tokenizer(
                            example["text"],
                            return_attention_mask=False,
                            truncation=True,
                            max_length=self.tokenizer_max_length
                        )["input_ids"]
                        for i in range(0, len(tokens) - self.chunk_length + 1, self.chunk_length):
                            chunk = tokens[i:i + self.chunk_length]
                            yield {
                                "input_ids": torch.tensor(chunk),
                                "attention_mask": torch.ones(len(chunk), dtype=torch.long)
                            }
                        count += 1

            self.dataset = WikipediaStreamDataset(
                self.tokenizer, config["max_len"], config.get("max_articles")
            )
        else:
            if config.get("use_streaming", False):
                from utils.web_streaming_dataset import WebCrawlStreamDataset
                print(f"[INFO] Using WebCrawlStreamDataset (live crawling)")
                self.dataset = WebCrawlStreamDataset(
                    urls=config["webcrawl_urls"],
                    tokenizer=self.tokenizer,
                    max_length=config["max_len"],
                    delay=config.get("crawl_delay", 1.0)
                )
            else:
                from utils.dataset import TextDataset
                print(f"[INFO] Using local text dataset from: {self.data_path}")
                self.dataset = TextDataset(str(self.data_path), self.tokenizer, max_length=config["max_len"])

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=not config.get("use_streaming", False),
            num_workers = (os.cpu_count() or 2) if not config.get("use_streaming", False) else 0
        )

        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

    def _build_model(self):
        dropout = self.config.get("dropout")
        model = GPTBackbone(
            vocab_size=self.config["vocab_size"],
            max_len=self.config["max_len"],
            embed_dim=self.config["embed_dim"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            dropout=dropout if dropout else 0.15
        )
        print(f"Model initialized with {count_parameters(model):,} parameters.")
        return model.to(self.device)

    def train_step(self, batch):
        if "input_ids" not in batch or "attention_mask" not in batch:
            raise ValueError("Missing input_ids or attention_mask in batch")

        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with autocast():
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Unscale gradients before measuring norm (AMP-safe)
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.tb_writer.add_scalar("GradNorm", grad_norm.item())

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.tb_writer.add_scalar("Loss/batch_train", loss.item())

        return loss.item()

    @torch.no_grad()
    def evaluate(self, num_batches=None):
        self.model.eval()
        total_loss = 0
        num_batches = num_batches or self.config.get("max_eval_batches") or 5
        progress_bar = tqdm(enumerate(self.dataloader), total=num_batches, desc="Evaluating", leave=False)

        for i, batch in progress_bar:
            if i >= num_batches:
                break
            try:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{(total_loss / (i + 1)):.4f}")
            except Exception as e:
                print(f"[WARN] Skipping eval batch {i} due to error: {e}")

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Validation — Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        return avg_loss, perplexity

    def train(self):
        print('')
        print(f"RUN TIME: {self.config['__run_time__']}")
        print('')
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            train_losses = []
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            max_batches = self.config.get("max_train_batches")
            total_batches = max_batches if max_batches is not None else None
            progress_bar = tqdm(enumerate(self.dataloader), total=total_batches, desc="Training", leave=False)

            for i, batch in progress_bar:
                if max_batches is not None and i >= max_batches:
                    break
                try:
                    loss = self.train_step(batch)
                    train_losses.append(loss)
                    progress_bar.set_postfix(loss=f"{(sum(train_losses) / len(train_losses)):.4f}")
                except Exception as e:
                    print(f"[WARN] Skipping batch {i} due to error: {e}")

            avg_train_loss = sum(train_losses) / len(train_losses)
            eval_loss, perplexity = self.evaluate()

            self.train_loss_history.append(avg_train_loss)
            self.eval_loss_history.append(eval_loss)
            self.lr_history.append(self.scheduler.get_last_lr()[0])

            print(f"Epoch {epoch} Complete — Train Loss: {avg_train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

            self.tb_writer.add_scalar("Loss/epoch_train", avg_train_loss, epoch)
            self.tb_writer.add_scalar("Loss/epoch_eval", eval_loss, epoch)
            self.tb_writer.add_scalar("Perplexity/epoch_eval", perplexity, epoch)
            self.tb_writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                ckpt_path = self.ckpt_dir / f"{self.config['__model_name__']}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": avg_train_loss,
                    "config": self.config,
                    "train_loss_history": self.train_loss_history,
                    "eval_loss_history": self.eval_loss_history,
                    "lr_history": self.lr_history,
                    "early_stop_state": self.early_stopping.__dict__
                }, ckpt_path)
                print(f"[INFO] Best model saved to {ckpt_path}")

                versioned_ckpt_path = self.ckpt_dir / f"{self.config['__model_name__']}_best_{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": avg_train_loss,
                    "config": self.config,
                    "train_loss_history": self.train_loss_history,
                    "eval_loss_history": self.eval_loss_history,
                    "lr_history": self.lr_history,
                    "early_stop_state": self.early_stopping.__dict__
                }, versioned_ckpt_path)

                metadata_path = self.ckpt_dir / f"{self.config['__model_name__']}_best_meta.json"
                with open(metadata_path, "w") as f:
                    json.dump({
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "eval_loss": eval_loss,
                        "perplexity": perplexity
                    }, f, indent=2)

            self.scheduler.step()

            # **Check for Early Stopping**
            if self.early_stopping.step(eval_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                # Should i save here the model again?
                break  # STOP TRAINING

        history_path = self.log_dir / "loss_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "train_loss": self.train_loss_history,
                "eval_loss": self.eval_loss_history,
                "lr": self.lr_history
            }, f)
        print(f"[INFO] Saved loss history to: {history_path}")

        csv_path = self.log_dir / "loss_history.csv"
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,eval_loss,lr\n")
            for i, (tr, ev, lr) in enumerate(
                    zip(self.train_loss_history, self.eval_loss_history, self.lr_history),
                    start=self.start_epoch
            ):
                f.write(f"{i},{tr},{ev},{lr}\n")
        print(f"[INFO] Saved loss history CSV to: {csv_path}")


def get_config(preset="base"):
    presets = {
        "micro":  {"embed_dim": 64,  "num_heads": 2,  "num_layers": 1},
        "tiny":   {"embed_dim": 128, "num_heads": 4,  "num_layers": 2},
        "small":  {"embed_dim": 192, "num_heads": 6,  "num_layers": 4},
        "base":   {"embed_dim": 256, "num_heads": 8,  "num_layers": 6},
        "medium": {"embed_dim": 512, "num_heads": 8,  "num_layers": 8},
        "large":  {"embed_dim": 512, "num_heads": 8,  "num_layers": 12},
        "xlarge": {"embed_dim": 768, "num_heads": 12, "num_layers": 24}
    }

    assert preset in presets, f"Invalid preset '{preset}'. Choose from: {list(presets.keys())}"

    return {
        "vocab_size": None,  # Let the model set this dynamically from the tokenizer
        "max_len": 256,
        "batch_size": 32,
        "dropout": 0.2,
        "lr": 1e-4,
        "max_articles": 5e4,  # Max number of Wikipedia articles to stream during training (via Hugging Face Datasets)
        "epochs": 10,
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
        "lr_gamma": 0.5,  # lr_step_size: 2 and lr_gamma: 0.5 means every 2 epochs the LR drops by half
        "resume_from_date": None,  # Provide a date to resume from
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
            "https://www.gutenberg.org/files/98/98-h/98-h.htm"
        ]
    }

if __name__ == "__main__":
    print("[WARN] Do not run this as a script directly. Use from Colab or external launcher.")
    import sys
    sys.exit(1)
