from torch.utils.data import IterableDataset
import requests
from bs4 import BeautifulSoup
import time


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def fetch_and_clean(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return clean_html(response.text)
    except Exception as e:
        print(f"[!] Failed to fetch {url}: {e}")
        return None


class WebCrawlStreamDataset(IterableDataset):
    def __init__(self, urls, tokenizer, max_length=64, delay=1.0):
        self.urls = urls
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.delay = delay
        self.failed_urls = set()

    def __iter__(self):
        successful = 0

        for url in self.urls:
            if url in self.failed_urls:
                print(f"[!] Skipping known bad URL: {url}")
                continue

            print(f"[*] Crawling: {url}")
            text = fetch_and_clean(url)
            if not text:
                self.failed_urls.add(url)
                continue

            print(f"[DEBUG] Text length for {url}: {len(text)}")

            # Tokenize full text (no truncation)
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=False
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            total_len = input_ids.size(0)

            if total_len < self.max_length:
                print(f"[!] Skipping short content from: {url}")
                self.failed_urls.add(url)
                continue

            # Yield chunks of max_length tokens
            num_chunks = 0
            for i in range(0, total_len - self.max_length + 1, self.max_length):
                chunk_ids = input_ids[i:i + self.max_length]
                chunk_mask = attention_mask[i:i + self.max_length]

                yield {
                    "input_ids": chunk_ids,
                    "attention_mask": chunk_mask
                }
                num_chunks += 1

            print(f"[✓] Yielded {num_chunks} samples from: {url}")
            time.sleep(self.delay)
            successful += 1

        if successful == 0:
            print("[!] WARNING: No successful crawls this epoch.")


