#!/usr/bin/env python3
"""Safe cleanup script for models folder"""

import os
from pathlib import Path
from datetime import datetime
import fnmatch

# Files to always keep
KEEP_FILES = ['best.pt', 'bc_best_history.json']

def get_file_info(filepath):
    """Get file info"""
    try:
        stat = filepath.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        return {
            'name': filepath.name,
            'size_mb': round(size_mb, 2),
            'modified': mtime.strftime('%Y-%m-%d %H:%M'),
        }
    except:
        return None

def should_delete(filepath):
    """Check if file should be deleted"""
    name = filepath.name
    
    # Always keep certain files
    if name in KEEP_FILES:
        return False
    
    # Check patterns
    patterns = [
        '*.pkl', 
        '*_distilled.pt',
        '*_latest.pt',
        '_*.pt',
        '*_gate.pt',
        '*_legacy.pt',
    ]
    
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    
    return False

def main():
    print("="*60)
    print("MODEL CLEANUP SCRIPT")
    print("="*60)
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("[ERROR] models/ directory not found!")
        return 1
    
    files_to_delete = []
    files_to_keep = []
    total_size = 0.0
    
    for f in sorted(models_dir.iterdir(), key=lambda x: x.stat().st_size, reverse=True):
        if f.is_file():
            info = get_file_info(f)
            if not info:
                continue
            
            total_size += info['size_mb']
            
            if should_delete(f):
                files_to_delete.append(info)
            else:
                files_to_keep.append(info)
    
    delete_size = sum(f['size_mb'] for f in files_to_delete)
    
    print(f"\nTotal files: {len(files_to_delete) + len(files_to_keep)}")
    print(f"Current size: {total_size:.2f} MB")
    
    print(f"\nFiles to DELETE ({len(files_to_delete)}):")
    for info in files_to_delete:
        print(f"  [DEL] {info['name']:50s} {info['size_mb']:8.2f} MB")
    
    print(f"\nFiles to KEEP ({len(files_to_keep)}):")
    for info in files_to_keep:
        marker = " <-- ESSENTIAL" if info['name'] in KEEP_FILES else ""
        print(f"  [OK ] {info['name']:50s} {info['size_mb']:8.2f} MB{marker}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Will delete: {delete_size:.2f} MB ({len(files_to_delete)} files)")
    print(f"  Will save: {total_size - delete_size:.2f} MB ({len(files_to_keep)} files)")
    print(f"  Space freed: {delete_size:.2f} MB ({delete_size/total_size*100:.1f}%)")
    
    print("\n" + "="*60)
    response = input("Proceed with cleanup? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Cancelled.")
        return 0
    
    print("\nDeleting files...")
    deleted_count = 0
    deleted_size = 0.0
    
    for info in files_to_delete:
        filepath = models_dir / info['name']
        try:
            filepath.unlink()
            deleted_count += 1
            deleted_size += info['size_mb']
            print(f"  [DELETED] {info['name']}")
        except Exception as e:
            print(f"  [ERROR] Could not delete {info['name']}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Cleanup complete!")
    print(f"  Deleted: {deleted_count} files, {deleted_size:.2f} MB")
    print(f"  Remaining: {len(files_to_keep)} files, {total_size - deleted_size:.2f} MB")
    
    return 0

if __name__ == "__main__":
    exit(main())
