#!/usr/bin/env python3
"""
Clear All Training Data Script

This script safely removes all ML training data including:
- Model checkpoints
- TensorBoard logs
- Training videos
- ML experiment database
- Python cache files

Use this when you want to start fresh with new training runs.
"""

import os
import shutil
from pathlib import Path
import sqlite3


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        print(f"Warning: Could not calculate size for {path}: {e}")
    return total


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_database_info(db_path: Path) -> dict:
    """Get information about the ML database."""
    if not db_path.exists():
        return {"exists": False}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get run count
        cursor.execute("SELECT COUNT(*) FROM experiment_runs")
        run_count = cursor.fetchone()[0]
        
        # Get metric count
        cursor.execute("SELECT COUNT(*) FROM training_metrics")
        metric_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "exists": True,
            "size": db_path.stat().st_size,
            "runs": run_count,
            "metrics": metric_count
        }
    except Exception as e:
        return {
            "exists": True,
            "size": db_path.stat().st_size,
            "error": str(e)
        }


def clear_directory(path: Path, description: str) -> bool:
    """Clear a directory while preserving .gitkeep files."""
    if not path.exists():
        print(f"  ⏭️  {description}: Directory doesn't exist, skipping")
        return True
    
    try:
        size = get_directory_size(path)
        file_count = sum(1 for _ in path.rglob('*') if _.is_file())
        
        print(f"  🗑️  {description}:")
        print(f"      Files: {file_count}")
        print(f"      Size: {format_size(size)}")
        
        # Remove all contents except .gitkeep
        for item in path.iterdir():
            if item.name == '.gitkeep':
                continue
            
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        print(f"      ✅ Cleared")
        return True
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False


def clear_database(db_path: Path) -> bool:
    """Clear the ML experiment database."""
    if not db_path.exists():
        print(f"  ⏭️  Database: File doesn't exist, skipping")
        return True
    
    try:
        info = get_database_info(db_path)
        
        print(f"  🗑️  ML Experiment Database:")
        print(f"      Path: {db_path}")
        print(f"      Size: {format_size(info['size'])}")
        if 'runs' in info:
            print(f"      Runs: {info['runs']}")
            print(f"      Metrics: {info['metrics']}")
        
        db_path.unlink()
        print(f"      ✅ Deleted")
        return True
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False


def clear_pycache(root_path: Path) -> bool:
    """Clear Python cache files."""
    try:
        cache_dirs = list(root_path.rglob('__pycache__'))
        pyc_files = list(root_path.rglob('*.pyc'))
        
        total_size = sum(get_directory_size(d) for d in cache_dirs)
        total_size += sum(f.stat().st_size for f in pyc_files if f.exists())
        
        print(f"  🗑️  Python Cache:")
        print(f"      Directories: {len(cache_dirs)}")
        print(f"      .pyc files: {len(pyc_files)}")
        print(f"      Size: {format_size(total_size)}")
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        
        for pyc_file in pyc_files:
            if pyc_file.exists():
                pyc_file.unlink()
        
        print(f"      ✅ Cleared")
        return True
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False


def main():
    """Main cleanup function."""
    print("=" * 70)
    print("🧹 ML Training Data Cleanup")
    print("=" * 70)
    print()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"📁 Project Root: {project_root}")
    print()
    
    # Calculate total size before cleanup
    print("📊 Analyzing current data...")
    print()
    
    directories_to_clear = [
        (project_root / "models" / "checkpoints", "Model Checkpoints"),
        (project_root / "logs" / "tb", "TensorBoard Logs"),
        (project_root / "video" / "milestones", "Milestone Videos"),
        (project_root / "video" / "eval", "Evaluation Videos"),
        (project_root / "video" / "render" / "parts", "Render Parts"),
    ]
    
    db_path = project_root / "ml_experiments.db"
    
    # Show what will be deleted
    total_size = 0
    for dir_path, _ in directories_to_clear:
        if dir_path.exists():
            total_size += get_directory_size(dir_path)
    
    if db_path.exists():
        total_size += db_path.stat().st_size
    
    print(f"💾 Total data to be cleared: {format_size(total_size)}")
    print()
    
    # Confirm deletion
    print("⚠️  WARNING: This will permanently delete all training data!")
    print("   - All model checkpoints will be lost")
    print("   - All training logs will be removed")
    print("   - All videos will be deleted")
    print("   - The experiment database will be cleared")
    print()
    
    response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print()
        print("❌ Cleanup cancelled")
        return
    
    print()
    print("🚀 Starting cleanup...")
    print()
    
    # Clear directories
    success_count = 0
    for dir_path, description in directories_to_clear:
        if clear_directory(dir_path, description):
            success_count += 1
        print()
    
    # Clear database
    if clear_database(db_path):
        success_count += 1
    print()
    
    # Clear Python cache (optional)
    clear_pycache(project_root / "tools" / "retro_ml_desktop")
    print()
    
    # Summary
    print("=" * 70)
    print("✅ Cleanup Complete!")
    print("=" * 70)
    print()
    print(f"📊 Summary:")
    print(f"   - Cleared {success_count}/{len(directories_to_clear) + 1} locations")
    print(f"   - Freed up approximately {format_size(total_size)}")
    print()
    print("🎯 You can now start fresh training runs!")
    print()


if __name__ == "__main__":
    main()

