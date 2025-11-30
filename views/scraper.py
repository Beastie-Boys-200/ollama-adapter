from typing import List
from controllers.web_parsing.types import Article
from controllers.web_parsing.scraper import extract_texts_from_links
from .parse import get_search_links


def search_and_extract(
    query: str, count: int = 10, with_log: bool = False
) -> List[Article]:
    """
    Searches for articles based on a query and extracts their text content.

    Args:
        query (str): Search term (e.g., "docker").
        count (int): Number of links to process.
        with_log (bool): If True, prints logs during extraction.

    Returns:
        List[Article]: A list of dictionaries containing "link" and "text" keys.
    """
    links = get_search_links(query, count)
    result = extract_texts_from_links(links, None, with_log)
    return result


def extract(
    links: List[str], drop_tags: List[str] = None, with_log: bool = False
) -> List[Article]:
    """
    Extracts text content from a given list of links.

    Args:
        links (List[str]): A list of URLs to extract content from.
        drop_tags (List[str]): Unique tags to remove from HTML.
        with_log (bool): If True, prints logs during extraction.

    Returns:
        List[Article]: A list of dictionaries containing "link" and "text" keys.
    """
    result = extract_texts_from_links(links, drop_tags, with_log)
    return result


if __name__ == "__main__":
    result = search_and_extract("roblox", with_log=True)
    print(result[0]["text"][:300])
