"""A tool that fetches web pages.

Guard rails, applied to the first URL and to every redirect:

* only ``http`` and ``https``;
* an optional host allow-list;
* the host must not resolve to a private, loopback, link-local or otherwise
  non-public address, unless ``allow_private=True``. An IPv6 address that
  embeds an IPv4 address (IPv4-mapped, IPv4-compatible, or the NAT64 prefix
  ``64:ff9b::/96``) is judged by the IPv4 address inside it.

The checks look at the exact host name the HTTP client connects to (its
ASCII form, for internationalised names).

Without ``allow_hosts``, each fetch asks for approval: a GET can carry data
out of the process in its URL, so an agent that can also read files could
otherwise send them anywhere. With ``allow_hosts`` the tool is read-only and
runs without asking.

The body is read up to ``max_bytes``, counted both as received and after
decompression; responses with more than one ``Content-Encoding`` layer, or
an encoding other than gzip or deflate, are refused.

Limit of the address check: the name is resolved once here and again by the
HTTP client when it connects, so a hostile DNS server could answer the two
lookups differently. A NAT64 prefix chosen by the network operator cannot be
recognised either. When fetched content or URLs come from untrusted parties,
also restrict outbound traffic at the network level.

Whatever a page says is data for the model to read, never instructions.
"""

from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import socket
import zlib
from collections.abc import Iterable
from html.parser import HTMLParser

import httpx

from ..errors import ToolError
from ..tools import Tool, tool

_REDIRECTS = {301, 302, 303, 307, 308}
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")
_SITE_LOCAL = ipaddress.ip_network("fec0::/10")


def web_tools(
    *,
    allow_hosts: Iterable[str] | None = None,
    allow_private: bool = False,
    max_bytes: int = 1_000_000,
    max_chars: int = 40_000,
    timeout: float = 20.0,
    max_redirects: int = 5,
    http_client: httpx.AsyncClient | None = None,
    requires_approval: bool | None = None,
) -> list[Tool]:
    """A ``fetch_url`` tool.

    Args:
        allow_hosts: Shell-style patterns of the host names that may be fetched,
            for example ``["docs.example.com", "*.wikipedia.org"]``. Default: any host.
        allow_private: Also fetch hosts with private, loopback or link-local addresses.
        max_bytes: The most bytes of a body that are read, before and after decompression.
        max_chars: The most characters of text returned.
        timeout: Seconds for each request.
        max_redirects: How many redirects are followed.
        http_client: The client to send requests with. Default: a new one per call.
        requires_approval: Ask before each fetch. Default: ask unless ``allow_hosts``
            is given. A tool that asks is not read-only, so ``Permissions(mode="read_only")``
            refuses it; ``False`` makes it read-only and run without asking.
    """
    patterns = tuple(allow_hosts) if allow_hosts is not None else None
    ask = patterns is None if requires_approval is None else requires_approval

    async def check(url: httpx.URL) -> None:
        if url.scheme not in ("http", "https") or not url.raw_host:
            raise ToolError("Only http:// and https:// URLs can be fetched.")
        # The name httpx connects to: IDNA 2008 A-labels, never re-encoded on the way.
        host = url.raw_host.decode("ascii").lower()
        names = {host, url.host.lower()}
        if patterns is not None and not any(
            fnmatch.fnmatchcase(name, pattern) for name in names for pattern in patterns
        ):
            raise ToolError(f"Fetching from {host} is not allowed here.")
        if allow_private:
            return
        port = url.port or (443 if url.scheme == "https" else 80)
        try:
            found = await asyncio.get_running_loop().getaddrinfo(
                host, port, type=socket.SOCK_STREAM
            )
        except OSError as error:
            raise ToolError(f"Could not resolve {host}: {error}") from error
        for *_, address in found:
            if not _is_public(address[0]):
                raise ToolError(f"{host} resolves to a non-public address and was not fetched.")

    @tool(
        read_only=not ask,
        idempotent=True,
        requires_approval=ask,
        timeout=timeout * (max_redirects + 1) + 5,
    )
    async def fetch_url(url: str) -> str:
        """Fetch a web page or text resource and return its readable text.

        Args:
            url: Absolute http(s) URL.
        """
        try:
            current = httpx.URL(url)
        except httpx.InvalidURL as error:
            raise ToolError(f"That is not a valid URL: {error}") from error
        client = http_client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        try:
            for _ in range(max_redirects + 1):
                await check(current)
                try:
                    async with client.stream(
                        "GET",
                        current,
                        headers={"Accept-Encoding": "gzip, deflate"},
                        follow_redirects=False,
                    ) as response:
                        if response.status_code in _REDIRECTS and "location" in response.headers:
                            current = current.join(response.headers["location"])
                            continue
                        body = await _read_body(response, max_bytes)
                        status = response.status_code
                        content_type = response.headers.get("content-type", "")
                        encoding = response.encoding or "utf-8"
                except (httpx.HTTPError, httpx.InvalidURL) as error:
                    raise ToolError(f"Fetching {current} failed: {error}") from error
                try:
                    text = body.decode(encoding, errors="replace")
                except LookupError:  # the server named an encoding Python does not know
                    text = body.decode("utf-8", errors="replace")
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


def _is_public(address: str) -> bool:
    """Whether an address returned by the resolver is a public one."""
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped.is_global
        if ip in _NAT64:  # translated to the IPv4 address in its last 32 bits
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF).is_global
        if ip in _IPV4_COMPATIBLE or ip in _SITE_LOCAL:  # deprecated forms
            return False
    return ip.is_global


async def _read_body(response: httpx.Response, max_bytes: int) -> bytes:
    """At most ``max_bytes`` of the body, decompressed in bounded steps.

    httpx's own decoding inflates every chunk without a limit, so a few hundred
    bytes of nested gzip could allocate gigabytes. Here the raw bytes are read
    and one gzip or deflate layer (with a zlib header or raw, as httpx reads it)
    is undone with an output limit.

    Raises:
        ToolError: The encoding is not one this tool reads, or the data is corrupt.
    """
    if response.is_stream_consumed:  # a transport that read (and decoded) the body already
        return response.content[:max_bytes]
    declared = ",".join(response.headers.get_list("content-encoding")).lower()
    layers = [part.strip() for part in declared.split(",") if part.strip() not in ("", "identity")]
    if len(layers) > 1 or (layers and layers[0] not in ("gzip", "x-gzip", "deflate")):
        raise ToolError(
            f"The server sent the body with Content-Encoding {', '.join(layers)}, "
            "which this tool does not read."
        )
    decoder = None
    if layers:
        # gzip expects a gzip header; deflate a zlib header (a gzip one is accepted too).
        decoder = zlib.decompressobj(zlib.MAX_WBITS | (32 if layers[0] == "deflate" else 16))
    # Some servers send "deflate" as raw deflate data with no header, as httpx accepts:
    # the bytes are kept until the first output, to start again without a header.
    start = bytearray() if layers == ["deflate"] else None
    body = bytearray()
    received = 0
    async for chunk in response.aiter_raw():
        received += len(chunk)
        if decoder is None:
            body += chunk[: max_bytes - len(body)]
        else:
            if start is not None:
                start += chunk
            try:
                try:
                    _inflate(decoder, chunk, body, max_bytes)
                except zlib.error:
                    if start is None or body:
                        raise
                    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
                    _inflate(decoder, bytes(start), body, max_bytes)
                    start = None
            except zlib.error as error:
                raise ToolError(
                    f"The body from {response.url} could not be decompressed."
                ) from error
            if body:
                start = None
        if len(body) >= max_bytes or received >= max_bytes:
            break
    return bytes(body)


def _inflate(decoder: zlib._Decompress, data: bytes, body: bytearray, max_bytes: int) -> None:
    """Add what ``data`` inflates to ``body``, never letting ``body`` exceed ``max_bytes``."""
    while data and len(body) < max_bytes and not decoder.eof:
        body += decoder.decompress(data, max_bytes - len(body))
        data = decoder.unconsumed_tail


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
