import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from tavily import TavilyClient

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError(
        "TAVILY_API_KEY is not set. "
        "Please add it to .env file to enable web search."
    )

mcp = FastMCP("Web-Search")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@mcp.tool()
def search_web(
    query: str = Field(..., description="Search query to find real-time information from the internet"),
    max_results: int = Field(5, description="Maximum number of web results to return (1-10)"),
    search_depth: str = Field(
        "basic",
        description="Search depth: 'basic' for fast lookup, 'advanced' for deeper research",
    ),
) -> str:
    """
    Search the internet for real-time information using Tavily Web Search API.
    Use this tool when:
    - `search_knowledge` returned no results from the local knowledge base.
    - The user explicitly asks for the latest/current/up-to-date information.
    - The question involves recent events, new releases, CVEs, or live data.
    This is a READ-ONLY tool and does not modify any data.
    """
    max_results = max(1, min(max_results, 10))

    try:
        response = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=False,
        )
    except Exception as e:
        return f"Web search failed: {str(e)}"

    results = response.get("results", [])
    if not results:
        return "No relevant web results found."

    formatted = []
    for idx, item in enumerate(results, 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "No snippet available.")
        score = item.get("score", 0.0)

        formatted.append(
            f"[{idx}] Web: {title} | URL: {url} (Score: {score:.2f})\n"
            f"Snippet: {content}"
        )

    return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    mcp.run()
