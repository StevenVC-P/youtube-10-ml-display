"""
Clean all training data, videos, and database records.
This script will delete:
- All model checkpoints
- All training outputs
- All videos
- The metrics database (metrics.db)
- The experiment tracking database (ml_experiments.db)
"""

import shutil
from pathlib import Path


def clean_all():
    """Clean all training data."""
    project_root = Path(__file__).parent

    items_to_delete = [
        project_root / "models" / "checkpoints",
        project_root / "outputs",
        project_root / "metrics.db",
        project_root / "ml_experiments.db",  # Experiment tracking database
    ]

    print("=" * 70)
    print("  Cleaning All Training Data")
    print("=" * 70)

    for item in items_to_delete:
        if item.exists():
            if item.is_dir():
                # Count items before deletion
                try:
                    item_count = sum(1 for _ in item.rglob("*"))
                    print(f"\n[DELETE] {item} ({item_count} items)")
                    shutil.rmtree(item)
                    print(f"  [OK] Deleted directory")
                except Exception as e:
                    print(f"  [ERROR] Failed to delete: {e}")
            else:
                # File
                try:
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"\n[DELETE] {item.name} ({size_mb:.2f} MB)")
                    item.unlink()
                    print(f"  [OK] Deleted file")
                except Exception as e:
                    print(f"  [ERROR] Failed to delete: {e}")
        else:
            print(f"\n[SKIP] {item.name} (doesn't exist)")

    # Recreate empty directories
    print("\n" + "=" * 70)
    print("  Recreating Empty Directories")
    print("=" * 70)

    dirs_to_create = [
        project_root / "models" / "checkpoints",
        project_root / "outputs",
    ]

    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[CREATE] {directory}")

    print("\n" + "=" * 70)
    print("  Cleanup Complete!")
    print("=" * 70)
    print("\nAll training data has been cleared. You can now start fresh testing.\n")


if __name__ == "__main__":
    clean_all()
