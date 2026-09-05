#!/usr/bin/env python3
import os
from pathlib import Path

def get_folder_size(path):
    """计算文件夹总大小 (字节)"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

folders = ['models', 'venv']

for folder in folders:
    path = Path(folder)
    if path.exists():
        total_bytes = get_folder_size(path)
        total_mb = total_bytes / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print(f"Folder: {folder}")
        print(f"{'='*60}")
        print(f"Total size: {total_mb:.2f} MB")
        
        # List top-level contents
        items = list(path.iterdir())
        print(f"\nTop-level items ({len(items)}):")
        
        for item in sorted(items, key=lambda x: x.stat().st_size, reverse=True)[:10]:
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  [FILE] {item.name}: {size_mb:.2f} MB")
            else:
                sub_size = get_folder_size(item) / (1024 * 1024)
                print(f"  [DIR ] {item.name}: {sub_size:.2f} MB")
                
    else:
        print(f"Folder '{folder}' does not exist!")

print("\n" + "="*60)
print("Recommendations:")
print("="*60)
print("1. models/ - 可能包含训练好的模型文件 (.pt/.pth)")
print("   → 可以删除不需要的模型，只保留 best.pt")
print()
print("2. venv/ - Python 虚拟环境 (500MB-2GB 是正常的)")
print("   → 不要删除！如果需要清理，可以重新创建")
print()
print("To delete old models:")
print("  rm -rf models/*.pt  (keep only needed ones)")
print("  rm -rf models/*.pth")
print()
print("To recreate venv:")
print("  python -m venv venv_new")
print("  source venv_new/bin/activate  # Windows: venv_new\\Scripts\\activate")
print("  pip install numpy pyyaml ...")
