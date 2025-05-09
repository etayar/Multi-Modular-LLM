from torch.utils.data import IterableDataset
from datasets import load_dataset


class StreamingTextDataset(IterableDataset):
    def __init__(self, dataset_name, tokenizer, max_length=64, split="train", dataset_config=None):
        if dataset_config is None:
            raise ValueError("You must provide `dataset_config` when using streaming=True")
        self.split = split
        self.dataset = load_dataset(
            path=dataset_name,
            name=dataset_config,
            split=self.split,
            streaming=True,
            trust_remote_code=True
        )

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
            yield encoding["input_ids"].squeeze(0)