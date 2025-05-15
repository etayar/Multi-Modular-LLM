import requests
from bs4 import BeautifulSoup
from pathlib import Path
import argparse
import time


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator=" ", strip=True)


def fetch_and_clean(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return clean_html(response.text)
    except Exception as e:
        print(f"[!] Failed to fetch {url}: {e}")
        return ""


def crawl(urls, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for url in urls:
            print(f"[*] Crawling: {url}")
            text = fetch_and_clean(url)
            if text:
                f.write(text + "\n")
            time.sleep(1)  # be polite
    print(f"Crawl complete. Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/web_corpus.txt")
    args = parser.parse_args()

    urls = [
        "https://en.wikipedia.org/wiki/Transformer_(machine_learning)",
        "https://en.wikipedia.org/wiki/History_of_mathematics",
        "https://en.wikipedia.org/wiki/Physics",
        "https://www.gutenberg.org/cache/epub/84/pg84.html",  # Frankenstein
        "https://www.gutenberg.org/cache/epub/98/pg98.html",  # A Tale of Two Cities
        # --- BBC News (Public Articles) ---
        "https://www.bbc.com/news/world-asia-68838588",  # News article
        "https://www.bbc.com/news/science-environment-68820146",
        "https://www.bbc.com/news/technology-68812335"
    ]

    crawl(urls, Path(args.output))


if __name__ == "__main__":
    main()
