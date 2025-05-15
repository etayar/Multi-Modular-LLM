import torch
import torch.nn as nn
import torch.optim as optim
from model import GPTBackbone
from model.utils import count_parameters
from utils.tokenizer import load_tokenizer
from torch.utils.data import DataLoader
from pathlib import Path
import csv
import subprocess
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter


class TransformerTrainer:
    def __init__(self, config):
        self.config = config
        self.config["__data_mode__"] = "streaming" if config.get("use_streaming", False) else "local"
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except Exception:
            commit_hash = "N/A"
        self.config["__git_commit__"] = commit_hash
        self.config["__run_time__"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.device = config["device"]
        self.tokenizer = load_tokenizer()
        true_vocab_size = self.tokenizer.vocab_size
        if "vocab_size" in self.config and self.config["vocab_size"] < true_vocab_size:
            print(f"[WARN] Config vocab_size ({self.config['vocab_size']}) is smaller than tokenizer vocab size ({true_vocab_size}). Updating.")
        self.config["vocab_size"] = true_vocab_size

        self.model = self._build_model()
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id set, which is required for ignore_index.")
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["lr"])

        self.project_root = Path(__file__).resolve().parents[1]
        self.ckpt_dir = self.project_root / config["ckpt_dir"]
        self.log_dir = self.project_root / config["log_dir"]
        self.data_path = self.project_root / config["dataset_path"]

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "train_log.csv"

        self.start_epoch = 1

        resume_path = config.get("resume_from")
        if resume_path:
            ckpt_path = Path(resume_path)
            if not ckpt_path.is_absolute():
                ckpt_path = self.ckpt_dir / ckpt_path
            if ckpt_path.exists():
                print(f"[INFO] Loading checkpoint from: {ckpt_path.name}")
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
                print(f"[INFO] Resuming from epoch {self.start_epoch}")
            else:
                print(f"[WARN] Specified checkpoint not found: {ckpt_path}")

        elif config.get("load_last_cp", False):
            ckpts = list(self.ckpt_dir.glob("*.pt"))
            if ckpts:
                latest_ckpt = sorted(ckpts)[-1]
                print(f"[INFO] Loading latest checkpoint: {latest_ckpt.name}")
                checkpoint = torch.load(latest_ckpt, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
                print(f"[INFO] Resuming from epoch {self.start_epoch}")

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
            num_workers=0
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
            dropout= dropout if dropout else 0.15
        )
        print(f"Model initialized with {count_parameters(model):,} parameters.")
        return model.to(self.device)

    def train_step(self, batch):

        if "input_ids" not in batch or "attention_mask" not in batch:
            raise ValueError("Missing input_ids or attention_mask in batch")

        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        logits = self.model(input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def save_checkpoint(self, epoch, loss):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.get("__model_name__", "model")
        dataset_name = self.config.get("dataset_name", "dataset")
        ckpt_path = self.ckpt_dir / f"{model_name}_{dataset_name}_epoch_{epoch:03d}_{timestamp}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    def log_metrics(self, epoch, train_loss, eval_loss, perplexity):
        self.config["__data_mode__"] = "streaming" if self.config.get("use_streaming", False) else "local"
        write_header = not self.log_path.exists()
        with self.log_path.open(mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "epoch", "train_loss", "eval_loss", "perplexity", "__data_mode__", "__git_commit__", "__run_time__", "__model_name__"] + list(self.config.keys()))
            writer.writerow([datetime.now().isoformat(), epoch, train_loss, eval_loss, perplexity, self.config["__data_mode__"], self.config["__git_commit__"], self.config["__run_time__"], self.config["__model_name__"]] + list(self.config.values()))

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
                avg_loss_so_far = total_loss / (i + 1)
                progress_bar.set_postfix(loss=f"{avg_loss_so_far:.4f}")
            except Exception as e:
                print(f"[WARN] Skipping eval batch {i} due to error: {e}")
                continue

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Validation — Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        return avg_loss, perplexity

    def train(self):
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            train_losses = []
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            max_batches = self.config.get("max_train_batches")
            total_batches = max_batches if max_batches is not None else None
            progress_bar = tqdm(enumerate(self.dataloader), total=total_batches, desc="Training", leave=False)
            print("[DEBUG] Starting training loop, waiting for first batch...")

            for i, batch in progress_bar:
                print(
                    f"[DEBUG] Got batch {i} with shape: input_ids={batch['input_ids'].shape}, attention_mask={batch['attention_mask'].shape}"
                )

                if max_batches is not None and i >= max_batches:
                    break
                try:
                    train_loss = self.train_step(batch)
                    train_losses.append(train_loss)
                    avg_train_loss = sum(train_losses) / len(train_losses)
                    progress_bar.set_postfix(loss=f"{avg_train_loss:.4f}")
                except Exception as e:
                    print(f"[WARN] Skipping batch {i} due to error: {e}")
                    continue

            avg_train_loss = sum(train_losses) / len(train_losses)
            eval_loss, perplexity = self.evaluate()
            print(f"Epoch {epoch} Complete — Train Loss: {avg_train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
            self.save_checkpoint(epoch, avg_train_loss)
            self.log_metrics(epoch, avg_train_loss, eval_loss, perplexity)
            self.scheduler.step()
            print(f"[DEBUG] LR after epoch {epoch}: {self.scheduler.get_last_lr()[0]:.6f}")

            self.tb_writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.tb_writer.add_scalar("Loss/eval", eval_loss, epoch)
            self.tb_writer.add_scalar("Perplexity/eval", perplexity, epoch)

        self.tb_writer.close()


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
        "vocab_size": 1000,
        "max_len": 64,
        "batch_size": 64,
        "dropout": 0.2,
        "lr": 1e-4,
        "epochs": 5,
        "log_dir": "logs",
        "ckpt_dir": "checkpoints",
        "dataset_path": "data/train.txt",
        "dataset_name": "webcrawl",
        "dataset_config": "20220301.en",
        "split": "train",
        "use_streaming": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_train_batches": None,
        "max_eval_batches": None,
        "lr_step_size": 2,
        "lr_gamma": 0.5,
        "load_last_cp": True,
        "resume_from": None,
        "__model_name__": "GPTBackbone",
        "crawl_delay": 1.0,
        **presets[preset],
        "webcrawl_urls": [
            "https://www.bbc.com/news/world-europe-68867750",
            "https://www.theguardian.com/world/2024/may/01/ukraine-war-frontline-report",
            "https://www.aljazeera.com/news/2024/5/14/gaza-ceasefire-latest",
            "https://arxiv.org/abs/2405.01700",
            "https://arxiv.org/abs/2405.00566",
            "https://www.gutenberg.org/files/11/11-h/11-h.htm",  # Alice in Wonderland
            "https://www.gutenberg.org/files/98/98-h/98-h.htm"    # A Tale of Two Cities
        ]
    }


if __name__ == "__main__":
    import sys
    print("[WARN] Do not run this as a script directly. Use from Colab or external launcher.")
    sys.exit(1)
