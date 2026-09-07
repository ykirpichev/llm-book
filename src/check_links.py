#!/usr/bin/env python3
"""Check Markdown publication links and fail on confirmed missing targets."""

from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


LINK_RE = re.compile(r"\]\((https://[^)]+)\)")
CONFIRMED_MISSING = {404, 410}


def discover(paths: list[Path]) -> list[str]:
    urls: set[str] = set()
    for path in paths:
        urls.update(LINK_RE.findall(path.read_text(encoding="utf-8")))
    return sorted(urls)


def check(url: str) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Engineering-LLM-Book-Link-Checker/1.0",
            "Range": "bytes=0-1023",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return url, str(response.status)
    except urllib.error.HTTPError as error:
        return url, str(error.code)
    except (urllib.error.URLError, TimeoutError) as error:
        return url, f"UNVERIFIED {error.reason if hasattr(error, 'reason') else error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("manuscript")])
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.glob("*.md")) if path.is_dir() else [path])
    urls = discover(files)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check, urls))

    missing = [(url, status) for url, status in results if status.isdigit() and int(status) in CONFIRMED_MISSING]
    unverified = [(url, status) for url, status in results if status.startswith("UNVERIFIED")]
    for url, status in results:
        print(f"{status:>12}  {url}")
    print(f"checked={len(results)} missing={len(missing)} unverified={len(unverified)}")
    if missing:
        raise SystemExit("FAIL: confirmed missing publication links")


if __name__ == "__main__":
    main()
