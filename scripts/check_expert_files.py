#!/usr/bin/env python3
"""
Expert Engine 快速文件检查脚本

检查所有核心文件是否存在且非空
"""

import os
from pathlib import Path

def main():
    print("\n" + "="*60)
    print("EXPERT ENGINE - FILE STRUCTURE CHECK")
    print("="*60)
    
    expert_dir = Path("junqi/expert")
    
    # 期望的文件列表
    expected_files = {
        "__init__.py": "Module initialization",
        "rule_validator.py": "Expert-0 Rule Validator",
        "tactical_analyzer.py": "Expert-1 Tactical Analyzer",
        "hidden_piece_belief.py": "Expert-2 Information Layer",
        "mobility_calculator.py": "Expert-2 Space Layer", 
        "tempo_tracker.py": "Expert-2 Tempo Layer",
        "threat_detection.py": "Expert-2 Threat System",
        "conditional_value.py": "Expert-2 Conditional Value",
        "search_optimizer.py": "Expert-3 Search Optimizer",
        "expert_engine.py": "Unified Interface",
    }
    
    results = []
    
    for filename, description in expected_files.items():
        filepath = expert_dir / filename
        
        if filepath.exists():
            size = filepath.stat().st_size
            lines = len(filepath.read_text().splitlines())
            
            if size > 100:  # Non-empty check
                status = "✓"
                results.append((filename, True, f"{lines} lines, {size} bytes"))
            else:
                status = "⚠️ "
                results.append((filename, False, f"Empty or too small"))
        else:
            status = "✗"
            results.append((filename, False, "File not found"))
        
        print(f"{status} {filename:30s} {description}")
        if status == "✓":
            print(f"   → {lines} lines, {size:,} bytes")
    
    # Summary
    print("\n" + "-"*60)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Files checked: {total}")
    print(f"Valid files: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All Expert Engine files present and valid!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} file(s) missing or invalid.")
        return 1

if __name__ == "__main__":
    exit(main())
