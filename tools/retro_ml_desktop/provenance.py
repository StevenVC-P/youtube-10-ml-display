import hashlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def sha256_file(path: Path) -> Optional[str]:
    """Return sha256 hex digest for a file, or None if missing."""
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None


def short_git_commit(project_root: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (short, full) git commit hashes for the repo at project_root.
    Returns (None, None) if git is unavailable.
    """
    try:
        full = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(project_root), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        short = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(project_root), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return short, full
    except Exception:
        return None, None


def sanitize_filename_component(value: str) -> str:
    """Sanitize a string for safe filenames while remaining human-readable."""
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip())
    return cleaned or "na"


def iso_utc_now() -> str:
    """ISO 8601 timestamp in UTC without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
