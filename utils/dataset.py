from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=64):
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        encoding = self.tokenizer(
            line,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        return encoding["input_ids"].squeeze(0)  # shape: [max_length]
