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


class TransformerTrainer:
    def __init__(self, config):
        print(f"[TransformerTrainer DEBUG] dataset_name = {config['dataset_name']}, dataset_config = {config.get('dataset_config')}")
        self.config = config
        self.config["__model_name__"] = "GPTBackbone"
        self.config["__data_mode__"] = "streaming" if config.get("use_streaming", False) else "local"
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except Exception:
            commit_hash = "N/A"
        self.config["__git_commit__"] = commit_hash
        self.config["__run_time__"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.device = config["device"]
        self.tokenizer = load_tokenizer()
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["lr"])

        self.project_root = Path(__file__).resolve().parents[1]
        self.ckpt_dir = self.project_root / config["ckpt_dir"]
        self.log_dir = self.project_root / config["log_dir"]
        self.data_path = self.project_root / config["dataset_path"]

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "train_log.csv"

        print(f"TransformerTrainer DEBUG, config-use_streaming: {config.get('use_streaming', False)}")
        if config.get("use_streaming", False):
            from utils.streaming_dataset import StreamingTextDataset
            print(f"[INFO] Using Hugging Face streaming dataset: {config['dataset_name']}")
            dataset_config = config["dataset_config"]
            split = config.get("split", "train")
            dataset_name = config["dataset_name"]
            print(f"[DEBUG] Loading dataset: {dataset_name} | config: {dataset_config} | split: {split}")
            self.dataset = StreamingTextDataset(
                dataset_name=dataset_name,
                tokenizer=self.tokenizer,
                max_length=config["max_len"],
                dataset_config=dataset_config,
                split=split
            )
        else:
            from utils.dataset import TextDataset
            print(f"[INFO] Using local text dataset from: {self.data_path}")
            self.dataset = TextDataset(str(self.data_path), self.tokenizer, max_length=config["max_len"])

        self.dataloader = DataLoader(self.dataset, batch_size=self.config["batch_size"], shuffle=not config.get("use_streaming", False))

    def _build_model(self):
        model = GPTBackbone(
            vocab_size=self.config["vocab_size"],
            max_len=self.config["max_len"],
            embed_dim=self.config["embed_dim"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"]
        )
        print(f"Model initialized with {count_parameters(model):,} parameters.")
        return model.to(self.device)

    def train_step(self, input_ids):
        self.model.train()
        logits = self.model(input_ids)
        loss = self.criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_checkpoint(self, epoch, loss):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:03d}_{timestamp}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config
        }, ckpt_path)
        print(f"✅ Checkpoint saved: {ckpt_path}")

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

        # Respect config override or default to 5
        num_batches = num_batches or self.config.get("max_eval_batches") or 5

        for i, input_ids in enumerate(self.dataloader):
            if i >= num_batches:
                break
            input_ids = input_ids.to(self.device)
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        return avg_loss, perplexity

    def train(self):
        for epoch in range(1, self.config["epochs"] + 1):
            train_losses = []
            for input_ids in self.dataloader:
                if self.config.get("max_train_batches") is not None and i >= self.config["max_train_batches"]:
                    break
                input_ids = input_ids.to(self.device)
                train_loss = self.train_step(input_ids)
                train_losses.append(train_loss)
            avg_train_loss = sum(train_losses) / len(train_losses)
            eval_loss, perplexity = self.evaluate()
            print(f"Epoch {epoch}/{self.config['epochs']} - Train Loss: {avg_train_loss:.4f} - Eval Loss: {eval_loss:.4f} - Perplexity: {perplexity:.2f}")
            self.save_checkpoint(epoch, avg_train_loss)
            self.log_metrics(epoch, avg_train_loss, eval_loss, perplexity)


def get_config():
    return {
        "vocab_size": 1000,
        "max_len": 64,
        "batch_size": 4,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "lr": 1e-4,
        "epochs": 5,
        "log_dir": "logs",
        "ckpt_dir": "checkpoints",
        "dataset_path": "data/train.txt",
        "dataset_name": "wikipedia",
        "dataset_config": "20220301.en",
        "split": "train",  # for quick testing, optional
        "use_streaming": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_train_batches": None,
        "max_eval_batches": None
    }


if __name__ == "__main__":
    import sys
    print("[WARN] Do not run this as a script directly. Use from Colab or external launcher.")
    sys.exit(1)

