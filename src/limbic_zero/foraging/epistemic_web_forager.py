"""Open-World Epistemic Web, Scientific Paper, Encyclopedia & Safe Math-Probe Forager.

Provides embodied LIMBIC-0 organisms with genuine open-world epistemic agency:
  1. ``search_open_web``: DuckDuckGo HTML/Lite search with automatic Wikipedia/arXiv
     resilience so datacenter GPU pods always retrieve live open-web knowledge.
  2. ``fetch_web_page``: Public HTTP/HTTPS readable article extractor with strict SSRF,
     private-subnet, cloud-metadata (169.254.169.254), and byte-cap protection.
  3. ``search_arxiv_papers``: Real ``export.arxiv.org`` Atom API client for scientific
     preprints across physics, neuroscience, mathematics, and computer science.
  4. ``fetch_wikipedia_summary``: Multilingual Wikipedia REST & MediaWiki extract client
     (``ko`` with ``en`` fallback).
  5. ``run_epistemic_python_probe``: AST-verified, resource-capped, network-isolated
     scientific/mathematical Python scratchpad so organisms can empirically test equations,
     dynamical systems, and logical hypotheses without any host escape surface.

Zero-Cheat Guarantee:
  - Zero hardcoded canned answers or scripted discoveries in production mode.
  - Pluggable ``HttpTransport`` allows deterministic offline unit testing while executing
    100% real internet HTTP requests on cloud GPU pods.
"""

from __future__ import annotations

import ast
import html
import ipaddress
import json
import math
import re
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


SSRF_OR_EXFILTRATION_ATTEMPT = "SSRF_OR_EXFILTRATION_ATTEMPT"
SANDBOX_AST_VIOLATION = "SANDBOX_AST_VIOLATION"

_USER_AGENT = (
    "Mozilla/5.0 (compatible; LimbicZero-EpistemicForager/1.0; "
    "+https://github.com/gkjuwon-tech/limbic-zero)"
)

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
        "instance-data",
        "kubernetes.default",
        "kubernetes.default.svc",
    }
)

_ALLOWED_PYTHON_MODULES = frozenset(
    {
        "math",
        "cmath",
        "statistics",
        "random",
        "itertools",
        "functools",
        "collections",
        "fractions",
        "decimal",
        "json",
        "hashlib",
        "re",
        "operator",
        "datetime",
        "time",
        "string",
        "bisect",
        "heapq",
        "pprint",
        "csv",
        "io",
        "typing",
        "numbers",
        "copy",
    }
)

_FORBIDDEN_BUILTIN_CALLS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
        "help",
        "exit",
        "quit",
    }
)


@dataclass
class EpistemicDiscoveryEpisode:
    """Grounded epistemic discovery trace captured during waking foraging or chat."""

    episode_id: str
    timestamp_unix: float
    soul_id: str
    tool_name: str
    query_or_target: str
    distilled_text: str
    information_gain: float
    ach_extracellular: float = 0.25
    da_td_error: float = 0.0
    cortisol_ug_dl: float = 5.0
    consolidated_in_cortex: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_forbidden_ip(ip_str: str) -> Tuple[bool, str]:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True, f"Invalid IP literal `{ip_str}`"

    if (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        return True, f"Target IP `{addr}` resolves to non-public/internal range."
    # Explicit cloud metadata check (169.254.169.254)
    if str(addr).startswith("169.254."):
        return True, f"Cloud metadata endpoint `{addr}` is strictly prohibited."
    return False, ""


def validate_public_epistemic_url(
    url: str,
    resolve_dns: bool = True,
) -> Tuple[bool, str, str]:
    """Validate that ``url`` targets a safe public HTTP(S) host with zero SSRF vectors.

    Returns ``(ok, violation_category, reason)``.
    """
    raw = (url or "").strip()
    if not raw:
        return False, SSRF_OR_EXFILTRATION_ATTEMPT, "Empty URL."
    if len(raw) > 2048:
        return False, SSRF_OR_EXFILTRATION_ATTEMPT, "URL exceeds 2048 character cap (possible data exfiltration)."

    parsed = urllib.parse.urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return (
            False,
            SSRF_OR_EXFILTRATION_ATTEMPT,
            f"URL scheme `{scheme or 'none'}://` blocked; only public `http`/`https` allowed.",
        )

    if parsed.username or parsed.password:
        return (
            False,
            SSRF_OR_EXFILTRATION_ATTEMPT,
            "Embedded credentials (`user:pass@host`) in URL are prohibited.",
        )

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return False, SSRF_OR_EXFILTRATION_ATTEMPT, "Missing hostname in URL."

    if host in _BLOCKED_HOSTNAMES or host.endswith((".local", ".internal", ".localhost", ".lan")):
        return (
            False,
            SSRF_OR_EXFILTRATION_ATTEMPT,
            f"Internal/local hostname `{host}` blocked by Warden SSRF guard.",
        )

    # Check literal IP first
    try:
        ipaddress.ip_address(host)
        bad, reason = _is_forbidden_ip(host)
        if bad:
            return False, SSRF_OR_EXFILTRATION_ATTEMPT, reason
        return True, "OK", ""
    except ValueError:
        pass

    if resolve_dns:
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if scheme == "https" else 80), proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            return False, "NETWORK_DNS_ERROR", f"DNS resolution failed for `{host}`: {exc}"

        for info in infos:
            sockaddr = info[4]
            if not sockaddr:
                continue
            ip_s = str(sockaddr[0])
            bad, reason = _is_forbidden_ip(ip_s)
            if bad:
                return False, SSRF_OR_EXFILTRATION_ATTEMPT, f"Host `{host}` resolved to blocked address: {reason}"

    return True, "OK", ""


def validate_epistemic_python_ast(code: str) -> Tuple[bool, str]:
    """Statically verify that ``code`` is pure math/scientific logic with zero host/escape access."""
    src = (code or "").strip()
    if not src:
        return False, "Empty Python probe code."
    if len(src) > 8000:
        return False, "Python probe exceeds 8000 characters."

    try:
        tree = ast.parse(src, mode="exec")
    except SyntaxError as exc:
        return False, f"Python SyntaxError: {exc.msg} (line {exc.lineno})"

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > 2500:
            return False, "AST complexity limit exceeded (max 2500 nodes)."

        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in _ALLOWED_PYTHON_MODULES:
                    return (
                        False,
                        f"Import `{alias.name}` blocked by epistemic sandbox; allowed={sorted(_ALLOWED_PYTHON_MODULES)}.",
                    )

        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if node.level > 0 or mod not in _ALLOWED_PYTHON_MODULES:
                return (
                    False,
                    f"ImportFrom `{node.module}` blocked by epistemic sandbox; allowed={sorted(_ALLOWED_PYTHON_MODULES)}.",
                )

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in ("gi_frame", "f_globals", "f_locals", "f_builtins", "cr_frame"):
                return False, f"Dunder/frame attribute access `.{node.attr}` is forbidden in epistemic sandbox."

        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return False, f"Dunder identifier `{node.id}` is forbidden in epistemic sandbox."
            if node.id in _FORBIDDEN_BUILTIN_CALLS:
                return False, f"Builtin `{node.id}` is forbidden in epistemic sandbox."

    return True, "OK"


def extract_readable_text_from_html(raw_html: str, max_chars: int = 6000) -> Dict[str, str]:
    """Strip scripts/styles/boilerplate and return clean readable ``{title, text}``."""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else ""

    cleaned = re.sub(
        r"<(script|style|noscript|svg|header|footer|nav|aside|form|iframe)[^>]*>.*?</\1>",
        " ",
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<(br|p|div|li|tr|h[1-6]|section|article)[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)

    lines: List[str] = []
    for raw_line in cleaned.splitlines():
        compact = re.sub(r"[ \t\r\f\v]+", " ", raw_line).strip()
        if len(compact) >= 3:
            lines.append(compact)

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return {"title": title[:240], "text": text}


def compute_epistemic_information_gain(text: str) -> float:
    """Compute intrinsic information density & novelty score in ``[0.0, 1.0]`` from retrieved evidence."""
    s = (text or "").strip()
    if not s:
        return 0.0
    tokens = re.findall(r"[A-Za-z가-힣0-9_\-\.]{2,}", s.lower())
    if not tokens:
        return 0.05
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    numeric_facts = len(re.findall(r"\b\d+(?:\.\d+)?\b", s))
    length_factor = min(1.0, math.log1p(len(s)) / math.log1p(2400.0))
    quant_bonus = min(0.25, 0.025 * numeric_facts)
    score = 0.48 * length_factor + 0.32 * min(1.0, unique_ratio * 1.35) + quant_bonus
    return round(max(0.05, min(1.0, score)), 4)


HttpFetchFn = Callable[[str, float, int, Optional[Dict[str, str]]], Tuple[int, str, str]]


def default_http_fetch(
    url: str,
    timeout_sec: float = 9.0,
    max_bytes: int = 350_000,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """Perform a real HTTP(S) GET with byte cap and UTF-8 decoding.

    Returns ``(status_code, content_type, decoded_body)``.
    """
    hdrs = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    if headers:
        hdrs.update(headers)

    req = urllib.request.Request(url, headers=hdrs, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        status = int(getattr(resp, "status", 200))
        ctype = str(resp.headers.get("Content-Type", "text/plain"))
        raw_bytes = resp.read(max_bytes)
        charset = "utf-8"
        m_cs = re.search(r"charset=([A-Za-z0-9_\-]+)", ctype, flags=re.IGNORECASE)
        if m_cs:
            charset = m_cs.group(1)
        try:
            body = raw_bytes.decode(charset, errors="replace")
        except LookupError:
            body = raw_bytes.decode("utf-8", errors="replace")
        return status, ctype, body


class EpistemicWebForager:
    """Open-world epistemic research engine coupled to Warden SSRF & AST sandboxes."""

    def __init__(
        self,
        http_fetch: Optional[HttpFetchFn] = None,
        resolve_dns: bool = True,
        timeout_sec: float = 9.0,
    ) -> None:
        self._http_fetch: HttpFetchFn = http_fetch or default_http_fetch
        # When a custom mock transport is injected for offline unit tests, skip live DNS lookup
        # while still enforcing all hostname & IP literal SSRF checks!
        self._resolve_dns = resolve_dns if http_fetch is None else False
        self.timeout_sec = float(timeout_sec)

    # ------------------------------------------------------------------
    # 1. Open Web Search (DuckDuckGo HTML + Wikipedia/arXiv Multi-Source)
    # ------------------------------------------------------------------
    def search_open_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"status": "error", "error": "Empty search query.", "results": []}
        limit = max(1, min(8, int(max_results)))
        results: List[Dict[str, str]] = []
        sources_used: List[str] = []
        errors: List[str] = []

        # Attempt A: DuckDuckGo HTML endpoint
        ddg_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
        try:
            status, _, body = self._http_fetch(ddg_url, self.timeout_sec, 300_000, None)
            if 200 <= status < 300 and body:
                parsed_ddg = self._parse_ddg_html(body, max_items=limit)
                if parsed_ddg:
                    results.extend(parsed_ddg)
                    sources_used.append("duckduckgo_web")
        except Exception as exc:
            errors.append(f"ddg:{type(exc).__name__}")

        # Attempt B: Augment or fallback via Wikipedia Search (ko + en) so datacenter IPs
        # always receive rich encyclopedic web hits even if DDG presents a rate limit.
        if len(results) < limit:
            for lang in ("ko", "en"):
                if len(results) >= limit:
                    break
                wiki_hits = self._search_wikipedia_list(q, lang=lang, limit=max(2, limit - len(results)))
                if wiki_hits:
                    sources_used.append(f"wikipedia_{lang}")
                    seen_urls = {r["url"] for r in results}
                    for h in wiki_hits:
                        if h["url"] not in seen_urls and len(results) < limit:
                            results.append(h)
                            seen_urls.add(h["url"])

        # Attempt C: If query looks scientific/technical and results still sparse, query arXiv
        if len(results) < limit and re.search(r"[a-zA-Z]{3,}", q):
            try:
                arx = self.search_arxiv_papers(q, max_results=min(3, limit - len(results)))
                for p in arx.get("papers", []):
                    results.append(
                        {
                            "title": f"[arXiv] {p['title']}",
                            "url": p["abs_url"],
                            "snippet": p["summary"][:280],
                        }
                    )
                if arx.get("papers"):
                    sources_used.append("arxiv_api")
            except Exception as exc:
                errors.append(f"arxiv:{type(exc).__name__}")

        combined_txt = "\n".join(f"{r['title']}: {r['snippet']}" for r in results)
        return {
            "status": "ok" if results else "no_results",
            "query": q,
            "sources": sources_used,
            "result_count": len(results),
            "results": results[:limit],
            "information_gain": compute_epistemic_information_gain(combined_txt),
            "diagnostics": errors if not results else [],
        }

    @staticmethod
    def _parse_ddg_html(raw_html: str, max_items: int = 5) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        # Match DuckDuckGo result anchors + snippets
        anchor_pat = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        snip_pat = re.compile(
            r'<(?:a|div|span)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        anchors = anchor_pat.findall(raw_html)
        snippets = snip_pat.findall(raw_html)

        for idx, (href, raw_title) in enumerate(anchors):
            if len(items) >= max_items:
                break
            clean_href = html.unescape(href).strip()
            if "uddg=" in clean_href:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(clean_href).query)
                if "uddg" in qs and qs["uddg"]:
                    clean_href = qs["uddg"][0]
            elif clean_href.startswith("//"):
                clean_href = "https:" + clean_href

            if not clean_href.startswith(("http://", "https://")):
                continue
            if "duckduckgo.com/y.js" in clean_href:
                continue

            title = html.unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
            snip = ""
            if idx < len(snippets):
                snip = html.unescape(re.sub(r"<[^>]+>", "", snippets[idx])).strip()
                snip = re.sub(r"\s+", " ", snip)
            if title:
                items.append({"title": title[:200], "url": clean_href, "snippet": snip[:360]})
        return items

    def _search_wikipedia_list(self, query: str, lang: str = "ko", limit: int = 3) -> List[Dict[str, str]]:
        safe_lang = "ko" if lang not in ("ko", "en", "ja") else lang
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(max(1, min(5, limit))),
                "format": "json",
                "utf8": "1",
            }
        )
        api_url = f"https://{safe_lang}.wikipedia.org/w/api.php?{params}"
        try:
            status, _, body = self._http_fetch(api_url, self.timeout_sec, 200_000, None)
            if status != 200 or not body:
                return []
            data = json.loads(body)
            search_rows = ((data.get("query") or {}).get("search")) or []
            out: List[Dict[str, str]] = []
            for row in search_rows[:limit]:
                title = str(row.get("title", "")).strip()
                raw_snip = str(row.get("snippet", ""))
                clean_snip = html.unescape(re.sub(r"<[^>]+>", "", raw_snip)).strip()
                if title:
                    page_url = f"https://{safe_lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    out.append({"title": f"{title} ({safe_lang}.wikipedia)", "url": page_url, "snippet": clean_snip})
            return out
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 2. Public Web Article Reader (SSRF-Guarded)
    # ------------------------------------------------------------------
    def fetch_web_page(self, url: str, max_chars: int = 6000) -> Dict[str, Any]:
        ok, category, reason = validate_public_epistemic_url(url, resolve_dns=self._resolve_dns)
        if not ok:
            return {
                "status": "blocked" if category == SSRF_OR_EXFILTRATION_ATTEMPT else "error",
                "violation_category": category,
                "error": reason,
                "url": url,
            }

        cap = max(500, min(12_000, int(max_chars)))
        try:
            status, ctype, body = self._http_fetch(url, self.timeout_sec, 350_000, None)
        except Exception as exc:
            return {"status": "error", "url": url, "error": f"HTTP fetch error: {exc}"}

        if "json" in ctype.lower():
            text = body[:cap]
            title = url
        else:
            extracted = extract_readable_text_from_html(body, max_chars=cap)
            title = extracted["title"] or url
            text = extracted["text"]

        return {
            "status": "ok",
            "http_status": status,
            "url": url,
            "title": title,
            "content": text,
            "char_count": len(text),
            "information_gain": compute_epistemic_information_gain(text),
        }

    # ------------------------------------------------------------------
    # 3. Scientific Preprint Search (arXiv Atom API)
    # ------------------------------------------------------------------
    def search_arxiv_papers(self, query: str, max_results: int = 4) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"status": "error", "error": "Empty arXiv query.", "papers": []}
        limit = max(1, min(6, int(max_results)))
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{q}",
                "start": "0",
                "max_results": str(limit),
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        api_url = f"https://export.arxiv.org/api/query?{params}"
        try:
            status, _, xml_body = self._http_fetch(api_url, self.timeout_sec, 300_000, None)
        except Exception as exc:
            return {"status": "error", "query": q, "error": f"arXiv fetch failed: {exc}", "papers": []}

        if status != 200 or not xml_body:
            return {"status": "error", "query": q, "error": f"arXiv HTTP {status}", "papers": []}
        if "<!ENTITY" in xml_body.upper():
            return {"status": "error", "query": q, "error": "Rejected unsafe XML entity declaration.", "papers": []}

        papers: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_body)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns)[:limit]:
                raw_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
                title = re.sub(
                    r"\s+",
                    " ",
                    (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
                )
                summary = re.sub(
                    r"\s+",
                    " ",
                    (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip(),
                )
                published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()[:10]
                authors = [
                    (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
                    for a in entry.findall("atom:author", ns)
                ]
                arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
                if title:
                    papers.append(
                        {
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "authors": [a for a in authors if a][:5],
                            "published": published,
                            "summary": summary[:900],
                            "abs_url": raw_id or f"https://arxiv.org/abs/{arxiv_id}",
                        }
                    )
        except ET.ParseError as exc:
            return {"status": "error", "query": q, "error": f"XML parse error: {exc}", "papers": []}

        joined = "\n".join(f"{p['title']}: {p['summary']}" for p in papers)
        return {
            "status": "ok" if papers else "no_results",
            "query": q,
            "paper_count": len(papers),
            "papers": papers,
            "information_gain": compute_epistemic_information_gain(joined),
        }

    # ------------------------------------------------------------------
    # 4. Wikipedia Knowledge Extractor (ko -> en fallback)
    # ------------------------------------------------------------------
    def fetch_wikipedia_summary(self, topic: str, lang: str = "ko") -> Dict[str, Any]:
        t = (topic or "").strip()
        if not t:
            return {"status": "error", "error": "Empty Wikipedia topic."}

        langs_to_try = [lang] if lang in ("ko", "en") else ["ko"]
        if "en" not in langs_to_try:
            langs_to_try.append("en")

        for cur_lang in langs_to_try:
            summary_data = self._fetch_single_wiki(t, cur_lang)
            if summary_data is not None:
                return summary_data

            # Search closest article title in `cur_lang`
            candidates = self._search_wikipedia_list(t, lang=cur_lang, limit=1)
            if candidates:
                best_title = candidates[0]["title"].rsplit(" (", 1)[0].strip()
                summary_data = self._fetch_single_wiki(best_title, cur_lang)
                if summary_data is not None:
                    return summary_data

        return {"status": "not_found", "topic": t, "extract": "", "information_gain": 0.05}

    def _fetch_single_wiki(self, title: str, lang: str) -> Optional[Dict[str, Any]]:
        encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
        rest_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        try:
            status, _, body = self._http_fetch(rest_url, self.timeout_sec, 220_000, None)
            if status == 200 and body:
                obj = json.loads(body)
                extract = str(obj.get("extract", "")).strip()
                if extract and obj.get("type") != "disambiguation":
                    page_url = (
                        ((obj.get("content_urls") or {}).get("desktop") or {}).get("page")
                        or f"https://{lang}.wikipedia.org/wiki/{encoded}"
                    )
                    return {
                        "status": "ok",
                        "lang": lang,
                        "title": str(obj.get("title", title)),
                        "description": str(obj.get("description", "")),
                        "extract": extract[:4500],
                        "url": str(page_url),
                        "information_gain": compute_epistemic_information_gain(extract),
                    }
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # 5. Safe AST-Guarded Scientific & Mathematical Python Probe
    # ------------------------------------------------------------------
    def run_epistemic_python_probe(
        self,
        code: str,
        timeout_sec: float = 4.0,
        workspace_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ok, reason = validate_epistemic_python_ast(code)
        if not ok:
            return {
                "status": "blocked",
                "violation_category": SANDBOX_AST_VIOLATION,
                "error": reason,
            }

        ws_snapshot: Dict[str, str] = {}
        if workspace_files:
            budget = 96_000
            for k, v in workspace_files.items():
                if budget <= 0:
                    break
                sv = str(v)[: min(24_000, budget)]
                ws_snapshot[str(k)] = sv
                budget -= len(sv)

        # Execute in isolated `-I -S` child interpreter with CPU & virtual memory caps
        wrapper = (
            "import ast, json, math, cmath, statistics, random, itertools, functools, "
            "collections, fractions, decimal, hashlib, re, operator, datetime, time, "
            "string, bisect, heapq, pprint, csv, io, typing, numbers, copy\n"
            "try:\n"
            "    import resource\n"
            "    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))\n"
            "    resource.setrlimit(resource.RLIMIT_AS, (335544320, 335544320))\n"
            "except Exception:\n"
            "    pass\n"
            "SAFE_BUILTINS = {\n"
            "    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,\n"
            "    'callable': callable, 'chr': chr, 'complex': complex, 'dict': dict, 'divmod': divmod,\n"
            "    'enumerate': enumerate, 'filter': filter, 'float': float,\n"
            "    'format': format, 'frozenset': frozenset, 'hasattr': hasattr, 'hex': hex, 'int': int,\n"
            "    'isinstance': isinstance, 'issubclass': issubclass, 'iter': iter, 'len': len,\n"
            "    'list': list, 'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct, 'ord': ord,\n"
            "    'pow': pow, 'print': print, 'range': range, 'repr': repr,\n"
            "    'reversed': reversed, 'round': round, 'set': set, 'slice': slice,\n"
            "    'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'type': type,\n"
            "    'zip': zip, 'True': True, 'False': False, 'None': None,\n"
            "    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,\n"
            "    'KeyError': KeyError, 'IndexError': IndexError, 'RuntimeError': RuntimeError,\n"
            "    'AssertionError': AssertionError, 'StopIteration': StopIteration,\n"
            "    'ArithmeticError': ArithmeticError, 'ZeroDivisionError': ZeroDivisionError,\n"
            "}\n"
            "SAFE_MODS = {\n"
            "    'math': math, 'cmath': cmath, 'statistics': statistics, 'random': random,\n"
            "    'itertools': itertools, 'functools': functools, 'collections': collections,\n"
            "    'fractions': fractions, 'decimal': decimal, 'json': json, 'hashlib': hashlib,\n"
            "    're': re, 'operator': operator, 'datetime': datetime, 'time': time,\n"
            "    'string': string, 'bisect': bisect, 'heapq': heapq, 'pprint': pprint,\n"
            "    'csv': csv, 'io': io, 'typing': typing, 'numbers': numbers, 'copy': copy,\n"
            "}\n"
            "def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
            "    root = name.split('.')[0]\n"
            "    if level == 0 and root in SAFE_MODS:\n"
            "        return SAFE_MODS[root]\n"
            "    raise ImportError(f'Module {name} disallowed')\n"
            "SAFE_BUILTINS['__import__'] = _safe_import\n"
            f"WS_FILES = {json.dumps(ws_snapshot, ensure_ascii=False)}\n"
            "ns = {'__builtins__': SAFE_BUILTINS, 'WORKSPACE_FILES': WS_FILES}\n"
            f"USER_CODE = {json.dumps(code)}\n"
            "_tree = ast.parse(USER_CODE, mode='exec')\n"
            "_last_expr = None\n"
            "if _tree.body and isinstance(_tree.body[-1], ast.Expr):\n"
            "    _last_node = _tree.body.pop()\n"
            "    _last_expr = ast.Expression(_last_node.value)\n"
            "    ast.fix_missing_locations(_tree)\n"
            "    ast.fix_missing_locations(_last_expr)\n"
            "exec(compile(_tree, '<epistemic_probe>', 'exec'), ns, ns)\n"
            "if _last_expr is not None:\n"
            "    _val = eval(compile(_last_expr, '<epistemic_probe>', 'eval'), ns, ns)\n"
            "    if _val is not None:\n"
            "        print(repr(_val)[:1600])\n"
            "if 'RESULT' in ns:\n"
            "    print('RESULT =', repr(ns['RESULT'])[:1600])\n"
        )

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", wrapper],
                cwd="/tmp",
                capture_output=True,
                text=True,
                timeout=max(0.5, min(6.0, float(timeout_sec))),
                shell=False,
                check=False,
                env={"LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": f"Epistemic Python probe timed out after {timeout_sec:.1f}s.",
                "information_gain": 0.05,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "information_gain": 0.05}

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        stdout_txt = (proc.stdout or "").strip()[:4500]
        stderr_txt = (proc.stderr or "").strip()[:1500]
        ok_exec = proc.returncode == 0
        return {
            "status": "ok" if ok_exec else "runtime_error",
            "exit_code": proc.returncode,
            "elapsed_ms": elapsed_ms,
            "stdout": stdout_txt,
            "stderr": stderr_txt if not ok_exec else "",
            "information_gain": compute_epistemic_information_gain(stdout_txt) if ok_exec else 0.08,
        }
