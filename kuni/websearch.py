"""Web search through the Ollama API. Port of WebSearch.cpp."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .config import config

log = logging.getLogger("WebSearch")
URL = "https://ollama.com/api/web_search"


class WebSearchError(RuntimeError):
    """A readable failure for the model; it never contains the API key or a response body dump."""


@dataclass
class Result:
    title: str
    url: str
    content: str


async def search(query: str, max_results: int = 0, http: httpx.AsyncClient | None = None) -> list[Result]:
    key = config().webSearchOllamaKey
    if not key:
        raise WebSearchError("web search is not configured (no Ollama key)")
    body: dict = {"query": query}
    if max_results > 0:
        body["max_results"] = max_results
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    client = http or httpx.AsyncClient(timeout=min(20.0, float(config().requestTimeoutSecs)))
    try:
        try:
            resp = await client.post(URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise WebSearchError(f"the search service is unreachable ({type(exc).__name__})") from None
    finally:
        if http is None:
            await client.aclose()
    if resp.status_code in (401, 403):
        raise WebSearchError("the search service rejected the key")
    if resp.status_code == 429:
        raise WebSearchError("the search service rate limit was reached; try again later")
    try:
        data = resp.json()
    except ValueError:
        raise WebSearchError(f"the search service returned an unexpected answer (HTTP {resp.status_code})") from None
    if isinstance(data, dict) and "error" in data:
        raise WebSearchError(f"search error: {str(data['error'])[:200]}")
    if not 200 <= resp.status_code < 300 or not isinstance(data, dict) or not isinstance(data.get("results", []), list):
        raise WebSearchError(f"the search service returned an unexpected answer (HTTP {resp.status_code})")
    return [Result(str(r.get("title", "")), str(r.get("url", "")), str(r.get("content", "")))
            for r in data.get("results", []) if isinstance(r, dict)]
