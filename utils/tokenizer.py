from transformers import AutoTokenizer

def load_tokenizer(name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token  # for batching
    return tokenizer
