from typing import List
from .types import Article
import os
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse, parse_qs, unquote


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def ddg_clean_link(raw_url: str) -> str:
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    parsed = urlparse(raw_url)
    qs = parse_qs(parsed.query)
    if "uddg" in qs:
        return unquote(qs["uddg"][0])
    return raw_url


def get_links_ddg(query: str, count: int) -> List[str]:
    if not (1 <= count <= 50):
        print("Error: Count must be between 1 and 50.")
        return []

    links = []
    query_enc = quote_plus(query)
    offset = 0

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    while len(links) < count:
        url = f"https://duckduckgo.com/html/?q={query_enc}&s={offset}"
        res = requests.get(url, headers=headers)
        print(res)

        soup = BeautifulSoup(res.text, "html.parser")
        items = soup.select(".result__a")

        if not items:
            break

        for a in items:
            href = a.get("href")
            if href:
                clean = ddg_clean_link(href)
                links.append(clean)
            if len(links) >= count:
                break

        offset += 50

    return links[:count]


def get_links_gfg(query: str, count: int) -> List[str]:
    if not (1 <= count <= 30):
        print("Error: Count must be between 1 and 30.")
        return []

    url = f"https://www.geeksforgeeks.org/search/?gq={query}"
    links = []

    while len(links) < count:
        res = requests.get(url)
        if res.status_code != 200:
            print(f"Error: {res.status_code}")
            break

        soup = BeautifulSoup(res.text, "html.parser")

        # Extract article links
        page_links = [a["href"] for a in soup.select("article a")]
        links.extend(page_links)

        # Update next URL
        next_btn = soup.find(
            "a",
            class_="PaginationContainer_paginationContainer__link__qTC3z",
            string="Next",
        )

        if not next_btn:
            break

        next_page = next_btn["href"]
        url = f"https://www.geeksforgeeks.org{next_page}"

    return links[:count]


def clean_html(html: str, drop_tags: List[str]) -> str:
    if not drop_tags:
        return html

    soup = BeautifulSoup(html, "html.parser")
    for tag in drop_tags:
        for element in soup.find_all(tag):
            element.decompose()

    return str(soup)


def extract_texts_from_links(
    links: List[str], drop_tags: List[str], with_log: bool
) -> List[Article]:
    if not (1 <= len(links) <= 30):
        print("Error: Link count must be between 1 and 30.")
        return []

    log_dir = "logs"
    if with_log:
        os.makedirs(log_dir, exist_ok=True)

    result = []

    # Extract article texts
    for i, link in enumerate(links, start=1):
        if with_log:
            print(f"[{i}/{len(links)}] Fetching: {link}")

        html = trafilatura.fetch_url(link)
        if not html:
            continue

        cleaned_html = clean_html(html, drop_tags)

        text = trafilatura.extract(cleaned_html)
        if not text:
            if with_log:
                print(f"{RED}FAILED{RESET}")
            continue

        result.append({"link": link, "text": text})
        if with_log:
            print(f"{GREEN}SUCCESS{RESET}")

    # Save texts
    if with_log:
        for i, entry in enumerate(result, start=1):
            with open(f"{log_dir}/text_{i}.txt", "w") as f:
                f.write(entry["text"])

    return result


if __name__ == "__main__":
    links = get_links_ddg("docker", count=5)
    print(links)
    print(extract_texts_from_links(links, drop_tags=None, with_log=False))
