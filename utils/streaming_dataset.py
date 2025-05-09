from torch.utils.data import IterableDataset
from datasets import load_dataset
import torch


class StreamingTextDataset(IterableDataset):
    def __init__(self, dataset_name, tokenizer, max_length=64, split="train"):
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for example in self.dataset:
            text = example.get("text")
            if not text:
                continue
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            yield encoding["input_ids"].squeeze(0)  # shape: [max_length]
