#!/usr/bin/env python3
"""
Stale run watchdog for Retro ML Desktop.

Marks database runs as failed when two conditions are met:
  1) process_pid is not running, and
  2) last activity (latest metric or updated_at) is older than the threshold.

Usable as a standalone CLI (cron/Task Scheduler) or imported and called from
the process manager loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import sys
import psutil
import sqlite3

# Ensure project root on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local import triggers schema migrations (e.g., status_note column)
from tools.retro_ml_desktop.ml_database import MetricsDatabase


@dataclass
class StaleRunResult:
    """Summary of a stale-run sweep."""
    checked: int
    marked_failed: int
    affected_runs: List[str]


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _pid_is_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    except Exception:
        return False


def _append_note(existing: Optional[str], note: str) -> str:
    if existing:
        return f"{existing.rstrip()}\n{note}"
    return note


def run_stale_run_guard(
    db_path: str,
    max_age_minutes: int = 30,
    require_pid: bool = True,
    mark_failed: bool = True,
    quiet: bool = False,
) -> StaleRunResult:
    """
    Evaluate running runs and mark obviously dead runs as failed.

    Two-signal default: pid missing AND last activity older than threshold.
    """
    MetricsDatabase(db_path)  # Ensures migrations run
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT run_id, process_pid, updated_at, description, status_note, start_time
        FROM experiment_runs
        WHERE status = 'running'
        """
    )
    rows = cursor.fetchall()

    now = datetime.now()
    cutoff = now - timedelta(minutes=max_age_minutes)
    marked = []

    for row in rows:
        run_id = row["run_id"]
        pid = row["process_pid"]
        updated_at = _parse_iso(row["updated_at"])

        cursor.execute(
            "SELECT timestamp FROM training_metrics WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (run_id,),
        )
        metric_row = cursor.fetchone()
        last_metric = _parse_iso(metric_row["timestamp"]) if metric_row else None

        last_activity = last_metric or updated_at or _parse_iso(row["start_time"])
        if not last_activity:
            # If timestamps are missing, treat as stale only when pid is gone
            last_activity = now - timedelta(days=365)

        pid_alive = _pid_is_alive(pid)
        old_enough = last_activity < cutoff

        should_mark = False
        signals = []
        if not pid_alive:
            signals.append("pid_missing")
        if old_enough:
            signals.append(f"last_activity>{max_age_minutes}m ({last_activity.isoformat()})")
        if require_pid:
            should_mark = (not pid_alive) and old_enough
        else:
            should_mark = (not pid_alive) or old_enough

        if not should_mark:
            continue

        reason = f"Auto-failed by stale-run-guard: {', '.join(signals)} at {now.isoformat()}"
        new_desc = _append_note(row["description"], reason)
        new_note = _append_note(row["status_note"], reason)

        if mark_failed:
            cursor.execute(
                """
                UPDATE experiment_runs
                SET status = 'failed',
                    end_time = ?,
                    description = ?,
                    status_note = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (now.isoformat(), new_desc, new_note, now.isoformat(), run_id),
            )
            conn.commit()

        marked.append(run_id)
        if not quiet:
            action = "Marked" if mark_failed else "Would mark"
            print(f"[stale-run-guard] {action} {run_id} as failed ({reason})")

    conn.close()
    return StaleRunResult(checked=len(rows), marked_failed=len(marked), affected_runs=marked)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and mark stale ML runs as failed.")
    parser.add_argument(
        "--db",
        default="ml_experiments.db",
        help="Path to ml_experiments.db (default: %(default)s)",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=int,
        default=30,
        help="Minutes since last activity before considering a run stale (default: %(default)s)",
    )
    parser.add_argument(
        "--any-signal",
        action="store_true",
        help="Mark as stale if either signal trips instead of requiring both.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write changes; just report what would happen.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    require_pid = not args.any_signal
    result = run_stale_run_guard(
        db_path=args.db,
        max_age_minutes=args.max_age_minutes,
        require_pid=require_pid,
        mark_failed=not args.dry_run,
        quiet=False,
    )

    if args.dry_run:
        print(
            f"[stale-run-guard] DRY RUN - checked {result.checked}, would mark {result.marked_failed}: {result.affected_runs}"
        )
    else:
        print(
            f"[stale-run-guard] checked {result.checked}, marked {result.marked_failed}: {result.affected_runs}"
        )


if __name__ == "__main__":
    main()
