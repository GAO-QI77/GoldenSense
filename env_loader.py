"""Minimal .env loader: repo-root file, real environment always wins.

No python-dotenv dependency. Lines are ``KEY=VALUE``; comments and blanks are
skipped; surrounding single/double quotes are stripped. A key already present
in ``os.environ`` is never overwritten, so shell exports and deployment
platform variables (Railway/Vercel/docker) take precedence over the file.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def load_env_file(path: str | os.PathLike = REPO_ROOT / ".env") -> int:
    """Load KEY=VALUE pairs from ``path`` into os.environ. Returns how many
    keys were set. Missing file is a silent no-op (the file is optional)."""
    path = Path(path)
    if not path.is_file():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded
