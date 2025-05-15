from torch.utils.data import IterableDataset
import requests
from bs4 import BeautifulSoup
import time
import random


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
        return ""


class WebCrawlStreamDataset(IterableDataset):
    def __init__(self, urls, tokenizer, max_length=64, delay=1.0, shuffle_each_epoch=True):
        self.all_urls = urls
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.delay = delay
        self.shuffle_each_epoch = shuffle_each_epoch
        self.failed_urls = set()

    def __iter__(self):
        urls = [u for u in self.all_urls if u not in self.failed_urls]
        if self.shuffle_each_epoch:
            random.shuffle(urls)

        for url in urls:
            print(f"[*] Crawling: {url}")
            text = fetch_and_clean(url)
            if not text:
                print(f"[!] Skipping: {url} (empty)")
                self.failed_urls.add(url)
                continue

            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

            input_ids = encoding.get("input_ids")
            attention_mask = encoding.get("attention_mask")
            if input_ids is None or attention_mask is None:
                print(f"[!] Tokenization failed: {url}")
                self.failed_urls.add(url)
                continue

            time.sleep(self.delay)

            yield {
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0)
            }
