#!/usr/bin/env python3
"""
Simple analysis script for Mini-Junqi experiment results
"""

import json
import sys
from pathlib import Path

def main():
    # Find latest report
    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("[ERROR] No reports found. Run an experiment first.")
        return 1
    
    reports = list(reports_dir.glob("mini_junqi_*.json"))
    if not reports:
        print("[ERROR] No mini_junqi reports found.")
        return 1
    
    latest_report = max(reports, key=lambda p: p.stat().st_mtime)
    
    print("="*60)
    print(f"Analyzing: {latest_report.name}")
    print("="*60)
    
    with open(latest_report, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Summary stats
    summary = data['summary']
    print("\n【基本信息】")
    print(f"总对局数：{summary['total_games']}")
    print(f"平均回合数：{summary['avg_turns']:.1f}")
    
    print("\n【胜率统计】")
    for color, rate in summary['win_rates'].items():
        percentage = rate * 100
        bar = "█" * int(percentage / 5)
        print(f"  {color.capitalize():7s}: {percentage:5.1f}% {bar}")
    
    # Win methods
    print("\n【获胜方式分布】")
    for method, count in sorted(data['detailed']['win_methods'].items(), 
                               key=lambda x: x[1], reverse=True):
        pct = count / summary['total_games'] * 100
        print(f"  {method}: {count} ({pct:.1f}%)")
    
    # Flip positions
    print("\n【Top 翻棋位置】")
    for i, (pos, count) in enumerate(list(data['detailed']['flip_positions'].items())[:5], 1):
        pct = count / summary['total_games'] * 100
        print(f"  {i}. {pos}: {count}次 ({pct:.1f}%)")
    
    # Insights
    print("\n【策略洞察】")
    if summary['avg_turns'] < 20:
        print("  → 游戏节奏较快，适合进攻型策略")
    elif summary['avg_turns'] < 35:
        print("  → 标准对局节奏")
    else:
        print("  → 持久战风格，防守策略更有效")
    
    red_win_rate = summary['win_rates']['red'] * 100
    black_win_rate = summary['win_rates']['black'] * 100
    if abs(red_win_rate - black_win_rate) > 15:
        winner = "Red" if red_win_rate > black_win_rate else "Black"
        print(f"  → {winner}方明显优势，建议检查平衡性")
    else:
        print("  → 双方实力接近，配置均衡")
    
    return 0

if __name__ == "__main__":
    exit(main())
