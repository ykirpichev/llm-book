#!/usr/bin/env python3
"""Conservative, offline credential/path scan of Git blobs; never print matches.

This is a release check, not a substitute for a dedicated secret scanner or
rights review. With --history it scans HEAD ancestry: the history actually
pushed with this branch. Local tool snapshots are intentionally not published.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader

PATTERNS = {
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "github-token": rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "aws-access-key": rb"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "service-token": rb"(?:sk-(?:proj-)?[A-Za-z0-9_-]{32,}|xox[baprs]-[A-Za-z0-9-]{20,})",
    "credential-in-url": rb"https?://[^\s/:]+:[^\s/@]+@",
    "personal-local-path": (rb"(?<![A-Za-z0-9./:])/(?:Users|home)/[A-Za-z0-9_.-]+/"
                            rb"|/var/" rb"folders/"),
}


def scan(data: bytes) -> list[str]:
    return [label for label, pattern in PATTERNS.items() if re.search(pattern, data)]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def audit(history: bool) -> int:
    if history:
        refs = git("rev-list", "--objects", "HEAD").splitlines()
    else:
        refs = [line.split(b"\t", 1)[0].split()[2] + b" " + line.split(b"\t", 1)[1]
                for line in git("ls-tree", "-r", "HEAD").splitlines()]
    seen: set[bytes] = set()
    scanned = images = findings = 0
    for entry in refs:
        parts = entry.split(b" ", 1)
        if len(parts) != 2:
            continue
        oid, name = parts
        if oid in seen or git("cat-file", "-t", oid.decode()).strip() != b"blob":
            continue
        seen.add(oid)
        path = name.decode(errors="replace")
        data = git("cat-file", "blob", oid.decode())
        if data.startswith(b"%PDF"):
            reader = PdfReader(io.BytesIO(data))
            data = "\n".join(p.extract_text() or "" for p in reader.pages).encode()
        elif Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            images += 1
            continue
        scanned += 1
        labels = scan(data)
        if labels:
            findings += 1
            print(f"REVIEW {oid.decode()[:12]} {path}: {', '.join(labels)}")
    print(f"scanned_blobs={scanned}; image_blobs_not_text_scanned={images}; flagged_blobs={findings}")
    return int(findings > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    raise SystemExit(audit(parser.parse_args().history))
