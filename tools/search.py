# tools/search.py

from tavily import TavilyClient

from config.settings import settings


tavily_client = TavilyClient(
    api_key=settings.TAVILY_API_KEY
)


def search_web(
    query: str,
    max_results: int = 5
) -> list[str]:
    """
    Search the web using Tavily
    and return URLs only.
    """

    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results
        )

        urls = []

        for result in response.get("results", []):
            url = result.get("url")

            if url:
                urls.append(url)

        return urls

    except Exception as e:
        print(f"Tavily search failed: {e}")
        return []