from torch.utils.data import IterableDataset
import requests
from bs4 import BeautifulSoup
import random
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
        return ""


class WebCrawlStreamDataset(IterableDataset):
    def __init__(self, urls, tokenizer, max_length=64, delay=1.0, shuffle_each_epoch=True):
        self.urls = urls
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.delay = delay
        self.shuffle_each_epoch = shuffle_each_epoch

    def __iter__(self):
        if self.shuffle_each_epoch:
            random.shuffle(self.urls[:])

        for url in self.urls[:]:
            print(f"[*] Crawling: {url}")
            text = fetch_and_clean(url)
            if not text:
                continue

            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            time.sleep(self.delay)
            yield {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            }
