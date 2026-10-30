"""
Scrape the English prose translation of every chapter (sarga) of Book One
(Bala Kanda) from valmikiramayan.net into a dict: {chapter_number: full_text}.

Usage:
    pip install requests beautifulsoup4
    python scrape_bala_kanda.py

Output: bala_kanda_english.json  (keys are chapter numbers as strings in JSON)
"""
import json
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://www.valmikiramayan.net/sundara/sundara_contents.htm"
OUT_FILE = "sundara_kanda_english.json"
DELAY = 1.0  # seconds between requests, be polite to the server

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (personal research scraper)"})


def get_soup(url, retries=3):
    """Fetch a URL and return BeautifulSoup, or None if it fails."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                # pass raw bytes so BeautifulSoup detects the encoding itself
                return BeautifulSoup(r.content, "html.parser")
            print(f"  HTTP {r.status_code} for {url}")
        except requests.RequestException as e:
            print(f"  error ({e}) for {url}")
        time.sleep(2 * (attempt + 1))
    return None


def get_chapter_links():
    """Return {chapter_number: frame_page_url} from the index page."""
    soup = get_soup(INDEX_URL)
    chapters = {}
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        a = tr.find("a", href=True)
        if len(cells) >= 2 and a:
            text = " ".join(cells[0].get_text().split())
            m = re.match(r"(\d+)", text)
            if m:
                chapters[int(m.group(1))] = urljoin(INDEX_URL, a["href"])
    return chapters


def find_prose_url(frame_url):
    """
    Each chapter link is a frameset page. Read its <frame src=...> tags and
    pick the one that is the English prose page (name contains 'prose').
    Falls back to guessing by swapping '_frame' for '_prose'.
    """
    soup = get_soup(frame_url)
    if soup:
        for tag in soup.find_all(["frame", "iframe", "a"]):
            src = tag.get("src") or tag.get("href")
            if src and "prose" in src.lower():
                return urljoin(frame_url, src)
    return frame_url.replace("_frame.htm", "_prose.htm")


def extract_text(url):
    soup = get_soup(url)
    if soup is None:
        return None
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = [" ".join(line.split()) for line in soup.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


def main():
    chapters = get_chapter_links()
    print(f"Found {len(chapters)} chapters")

    # resume support: keep anything already downloaded
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
            result = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        result = {}

    for num, frame_url in sorted(chapters.items()):
        if num in result:
            continue
        print(f"Chapter {num}")
        prose_url = find_prose_url(frame_url)
        time.sleep(DELAY)
        text = extract_text(prose_url)
        if text:
            result[num] = text
        else:
            print(f"  FAILED: {prose_url}")
        time.sleep(DELAY)

        # save progress after every chapter
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

    print(f"Done. {len(result)}/{len(chapters)} chapters saved to {OUT_FILE}")
    return result


if __name__ == "__main__":
    bala_kanda = main()
    # bala_kanda[1] -> full English text of chapter 1