"""A tool that fetches web pages.

Guard rails, applied to the first URL and to every redirect:

* only ``http`` and ``https``;
* an optional host allow-list;
* the host must not resolve to a private, loopback, link-local or otherwise
  non-public address, unless ``allow_private=True``.

Limit of the address check: the name is resolved once here and again by the
HTTP client when it connects, so a hostile DNS server could answer the two
lookups differently. When fetched content or URLs come from untrusted
parties, also restrict outbound traffic at the network level.

Whatever a page says is data for the model to read, never instructions.
"""

from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import socket
from collections.abc import Iterable
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from ..errors import ToolError
from ..tools import Tool, tool

_REDIRECTS = {301, 302, 303, 307, 308}


def web_tools(
    *,
    allow_hosts: Iterable[str] | None = None,
    allow_private: bool = False,
    max_bytes: int = 1_000_000,
    max_chars: int = 40_000,
    timeout: float = 20.0,
    max_redirects: int = 5,
    http_client: httpx.AsyncClient | None = None,
) -> list[Tool]:
    patterns = tuple(allow_hosts) if allow_hosts is not None else None

    async def check(url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ToolError("Only http:// and https:// URLs can be fetched.")
        host = parts.hostname.lower()
        if patterns is not None and not any(fnmatch.fnmatchcase(host, p) for p in patterns):
            raise ToolError(f"Fetching from {host} is not allowed here.")
        if allow_private:
            return
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            found = await asyncio.get_running_loop().getaddrinfo(
                host, port, type=socket.SOCK_STREAM
            )
        except OSError as error:
            raise ToolError(f"Could not resolve {host}: {error}") from error
        for *_, address in found:
            if not ipaddress.ip_address(address[0].split("%")[0]).is_global:
                raise ToolError(f"{host} resolves to a non-public address and was not fetched.")

    @tool(read_only=True, timeout=timeout * (max_redirects + 1) + 5)
    async def fetch_url(url: str) -> str:
        """Fetch a web page or text resource and return its readable text.

        Args:
            url: Absolute http(s) URL.
        """
        client = http_client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        try:
            current = url
            for _ in range(max_redirects + 1):
                await check(current)
                try:
                    async with client.stream("GET", current, follow_redirects=False) as response:
                        if response.status_code in _REDIRECTS and "location" in response.headers:
                            current = urljoin(current, response.headers["location"])
                            continue
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk[: max_bytes - len(body)])
                            if len(body) >= max_bytes:
                                break
                        status = response.status_code
                        content_type = response.headers.get("content-type", "")
                        encoding = response.encoding or "utf-8"
                except httpx.HTTPError as error:
                    raise ToolError(f"Fetching {current} failed: {error}") from error
                try:
                    text = bytes(body).decode(encoding, errors="replace")
                except LookupError:  # the server named an encoding Python does not know
                    text = bytes(body).decode("utf-8", errors="replace")
                if "html" in content_type:
                    text = html_to_text(text)
                if len(text) > max_chars:
                    text = f"{text[:max_chars]}\n[cut at {max_chars} characters]"
                if status >= 400:
                    raise ToolError(f"{current} answered HTTP {status}.\n{text[:2000]}")
                return f"URL: {current}\nContent-Type: {content_type}\n\n{text}"
            raise ToolError(f"Too many redirects (more than {max_redirects}).")
        finally:
            if http_client is None:
                await client.aclose()

    return [fetch_url]


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "svg"}
    _BREAK = {
        "p",
        "br",
        "div",
        "li",
        "tr",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pieces: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BREAK:
            self.pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BREAK:
            self.pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.pieces.append(data)


def html_to_text(html: str) -> str:
    """The visible text of an HTML document, one block per line."""
    extractor = _TextExtractor()
    extractor.feed(html)
    lines = (" ".join(line.split()) for line in "".join(extractor.pieces).splitlines())
    return "\n".join(line for line in lines if line)
