"""
Naming utilities for deterministic lineage-aware display names.

Display name format:
<root_name>-<leg_number><branch_token>-v<variant_index>

Branch tokens follow Excel-style sequencing: A, B, ..., Z, AA, AB, ...
"""

import string
from typing import Iterable


def _token_to_number(token: str) -> int:
    """Convert Excel-style token to 1-based number (A=1, B=2, ... Z=26, AA=27...)."""
    token = token.upper()
    value = 0
    for ch in token:
        if ch not in string.ascii_uppercase:
            return 0
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value


def _number_to_token(number: int) -> str:
    """Convert 1-based number to Excel-style token."""
    if number <= 0:
        return "A"
    token_chars = []
    n = number
    while n > 0:
        n, rem = divmod(n - 1, 26)
        token_chars.append(chr(ord("A") + rem))
    return "".join(reversed(token_chars))


def next_branch_token(existing_tokens: Iterable[str]) -> str:
    """
    Allocate the next branch token given existing tokens for a base_run_id.

    Tokens are treated as Excel columns. We pick max(existing)+1 to keep ordering stable.
    """
    max_number = 0
    for tok in existing_tokens:
        max_number = max(max_number, _token_to_number(tok or ""))
    return _number_to_token(max_number + 1)


def build_display_name(root_name: str, leg_number: int, branch_token: str, variant_index: int) -> str:
    """Construct the lineage display name."""
    safe_root = root_name.strip() if root_name else "run"
    branch = branch_token or "A"
    leg = leg_number if leg_number is not None else 1
    variant = variant_index if variant_index is not None else 1
    return f"{safe_root}-{leg}{branch}-v{variant}"


def sanitize_for_filename(name: str) -> str:
    """Basic filename sanitization for display names in artifacts."""
    allowed = "-._"
    return "".join(ch for ch in name if ch.isalnum() or ch in allowed) or "run"
