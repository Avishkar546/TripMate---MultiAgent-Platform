from tavily import TavilyClient
from config import settings

tavily_client = TavilyClient(api_key=settings.tavily_api_key)

def tavily_search(query):
    response = tavily_client.search(
        query=query,
        max_results=3
    )
    results = []

    for i, r in enumerate(response["results"]):
        title = r.get("title", "UNKNOWN")
        url = r.get("url", "")
        snippet = r.get("content", "").strip()
        published_date = r.get("published_date")

        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
        results.append(f"{i}. **{title}**\n  {url}\n  {snippet}\n  {published_date}")

    return "\n\n".join(results)