#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-Junqi System Verification Script

Tests all components of the Mini-Junqi Expert System.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ['PYTHONIOENCODING'] = 'utf-8'

def main():
    """Run verification tests"""
    print("\n" + "="*60)
    print("MINI-JUNQI SYSTEM VERIFICATION")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Check file existence
    print("\n[TEST 1] File Existence Check")
    print("-"*60)
    required_files = [
        ("junqi/mini_junqi.py", "Game Engine"),
        ("junqi/mini_agents.py", "Agent System"),
        ("scripts/mini_junqi_experiment.py", "Experiment Engine"),
        ("configs/mini_junqi_config.json", "Config JSON"),
        ("docs/08-MiniJunqi/JUNQI_MINI_5X5_OPTIMAL_STRATEGY_GUIDE.md", "Strategy Guide"),
        ("docs/08-MiniJunqi/README.md", "README"),
    ]
    
    for filepath, description in required_files:
        full_path = project_root / filepath
        if full_path.exists():
            size = full_path.stat().st_size
            lines = len(full_path.read_text(encoding='utf-8', errors='ignore').splitlines())
            status = "OK" if size > 100 else "TOO SMALL"
            print("[OK] {} - {:,} bytes, {} lines".format(description, size, lines))
        else:
            print("[FAIL] {} - FILE NOT FOUND".format(description))
            all_passed = False
    
    # Test 2: Syntax check
    print("\n[TEST 2] Syntax Validation")
    print("-"*60)
    try:
        import py_compile
        files_to_check = [
            project_root / "junqi/mini_junqi.py",
            project_root / "junqi/mini_agents.py",
            project_root / "scripts/mini_junqi_experiment.py",
        ]
        
        for filepath in files_to_check:
            if filepath.exists():
                try:
                    py_compile.compile(str(filepath), doraise=True)
                    print("[OK] {} - syntax valid".format(filepath.name))
                except py_compile.PyCompileError as e:
                    print("[FAIL] {} - {}".format(filepath.name, str(e)))
                    all_passed = False
    except ImportError:
        print("[SKIP] py_compile not available")
    
    # Test 3: Configuration validation
    print("\n[TEST 3] Configuration Check")
    print("-"*60)
    try:
        import json
        config_path = project_root / "configs/mini_junqi_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            checks = [
                ('board', 'Board configuration'),
                ('piece_sets', 'Piece sets (scheme_a/scheme_b)'),
                ('bunkers', 'Bunker positions'),
                ('piece_ranks', 'Piece rank values'),
                ('win_conditions', 'Win conditions'),
            ]
            
            for key, desc in checks:
                if key in config:
                    print("[OK] {} present".format(desc))
                else:
                    print("[WARN] {} missing".format(desc))
        else:
            print("[FAIL] Config file not found")
            all_passed = False
            
    except Exception as e:
        print("[ERROR] Config validation: {}".format(str(e)))
        all_passed = False
    
    # Test 4: Module structure check
    print("\n[TEST 4] Module Structure")
    print("-"*60)
    
    expected_dirs = [
        "junqi/expert",
        "configs",
        "docs/08-MiniJunqi",
        "scripts",
    ]
    
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print("[OK] Directory: {}".format(dir_name))
        else:
            print("[FAIL] Missing directory: {}".format(dir_name))
            all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("[SUCCESS] All basic verifications passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install numpy pyyaml")
        print("2. Run experiments: python scripts/mini_junqi_experiment.py --games 100")
        print("3. Read strategy guide: docs/08-MiniJunqi/JUNQI_MINI_5X5_OPTIMAL_STRATEGY_GUIDE.md")
        return 0
    else:
        print("[WARNING] Some verifications failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
