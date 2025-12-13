#!/usr/bin/env python3
"""
Hard reset utility for ml_experiments.db.

This script replaces the database file by renaming it to a timestamped backup, or deleting it if requested.
It does not perform row-by-row deletion. On next app start, the DB will be recreated empty by normal initialization.
"""

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Hard reset ml_experiments.db (destructive).")
    parser.add_argument(
        "--db-path",
        default="ml_experiments.db",
        help="Path to the database file (default: ml_experiments.db in current directory).",
    )
    parser.add_argument(
        "--yes-really",
        action="store_true",
        help="Required to actually perform the reset.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="If rename/backup fails, delete the DB instead.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = Path(args.db_path)
    backup_path = db_path.with_name(f"{db_path.name}.bak-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"[RESET-DB] Target DB: {db_path}")
    print(f"[RESET-DB] Planned backup: {backup_path}")
    if not args.yes_really:
        print("[RESET-DB] Dry run (no --yes-really provided). No action taken.")
        sys.exit(0)

    if not db_path.exists():
        print("[RESET-DB] DB file does not exist. Nothing to reset.")
        sys.exit(0)

    # Ensure no open connections in this process (none are created here).
    try:
        # Attempt backup via rename
        shutil.move(str(db_path), str(backup_path))
        print(f"[RESET-DB] Backup created: {backup_path}")
    except Exception as e:
        print(f"[RESET-DB] Backup rename failed: {e}")
        if args.delete:
            try:
                db_path.unlink(missing_ok=True)
                print(f"[RESET-DB] Deleted DB file: {db_path}")
            except Exception as del_err:
                print(f"[RESET-DB] Failed to delete DB file: {del_err}")
                sys.exit(1)
        else:
            print("[RESET-DB] No deletion performed (use --delete to allow delete fallback).")
            sys.exit(1)

    print("[RESET-DB] Reset complete. The next app start will create a new empty DB.")


if __name__ == "__main__":
    main()
