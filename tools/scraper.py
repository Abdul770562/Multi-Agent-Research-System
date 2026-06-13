# tools/scraper.py

import requests

from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent":
    "Mozilla/5.0"
}


def scrape_url(url: str) -> str:
    """
    Scrape webpage text.
    """

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header"
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return text[:5000]

    except Exception:
        return ""