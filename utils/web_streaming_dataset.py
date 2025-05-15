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
        for url in self.urls:
            if url in self.failed_urls:
                print(f"[!] Skipping known bad URL: {url}")
                continue

            print(f"[*] Crawling: {url}")
            text = fetch_and_clean(url)
            if not text:
                self.failed_urls.add(url)
                continue

            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

            if encoding["input_ids"].nelement() == 0:
                print(f"[!] Skipping empty encoding for: {url}")
                self.failed_urls.add(url)
                continue

            time.sleep(self.delay)

            yield {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            }
