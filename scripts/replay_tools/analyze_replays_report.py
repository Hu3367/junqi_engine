#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成军棋复盘分析报告"""

import sys
sys.path.insert(0, '.')

import json
from collections import Counter, defaultdict
from datetime import datetime

def analyze_replays(json_file):
    """深度分析所有复盘数据"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    games = data['games']
    stats = data['statistics']
    
    print("=" * 70)
    print("         军棋复盘数据分析报告")
    print("=" * 70)
    print()
    
    # 1. 总体概览
    print("【总体概览】")
    print("-" * 70)
    print("总复盘文件数:", stats['total_games'])
    print("成功解析数:", stats['valid_games'])
    print('成功率:', '{:.1f}%'.format(100*stats['valid_games']/stats['total_games']))
    print()
    print("对局模式分布:")
    for mode, count in sorted(stats['modes'].items()):
        pct = 100*count/stats['valid_games']
        bar = '=' * int(pct/2)
        print('  模式 {}: {:4d} 局 ({:.1f}%) {}'.format(mode, count, pct, bar))
    print()
    
    # 2. 着法数统计
    print("【着法数统计】")
    print("-" * 70)
    move_counts = [g["game_info"]["total_moves"] for g in games if not g.get("error")]
    move_counts_sorted = sorted(move_counts)
    
    def percentile(data, p):
        return data[int(p * len(data))]
    
    print("最小值:", min(move_counts))
    print("最大值:", max(move_counts))
    print("平均值: {:.1f}".format(sum(move_counts)/len(move_counts)))
    print("中位数:", percentile(move_counts_sorted, 0.5))
    print("P25:", percentile(move_counts_sorted, 0.25))
    print("P75:", percentile(move_counts_sorted, 0.75))
    print()
    
    # 3. 胜负分析
    print("【胜负分析】")
    print("-" * 70)
    winners = stats['winners']
    decided = winners['player1'] + winners['player2']
    
    print("已结束对局:", decided)
    print("未完成对局:", winners['undecided'])
    print("和棋:", winners['draw'])
    print()
    
    if decided > 0:
        p1_win_pct = 100*winners['player1']/decided
        p2_win_pct = 100*winners['player2']/decided
        print("先手胜率: {:.1f}% ({}/{})".format(p1_win_pct, winners['player1'], decided))
        print("后手胜率: {:.1f}% ({}/{})".format(p2_win_pct, winners['player2'], decided))
    print()
    
    # 4. 失败复盘分析
    failed_games = [g for g in games if g.get("error")]
    if failed_games:
        print("【失败复盘 ({}局)】".format(len(failed_games)))
        print("-" * 70)
        for fg in failed_games[:10]:
            print("  - {}: {}".format(fg['filename'], fg['error']))
        if len(failed_games) > 10:
            print("  ... 还有 {}个失败文件".format(len(failed_games) - 10))
        print()
    
    # 5. 长局分析（超过平均值的 2 倍）
    long_games = [g for g in games if not g.get("error") and g["game_info"]["total_moves"] > 200]
    if long_games:
        print("【超长对局 (>200 步) - {}局】".format(len(long_games)))
        print("-" * 70)
        for lg in sorted(long_games, key=lambda x: -x["game_info"]["total_moves"])[:5]:
            moves = lg["game_info"]["total_moves"]
            winner = "先手胜" if lg["result"]["winner"] == 0 else "后手胜" if lg["result"]["winner"] == 1 else "其他"
            print("  - {:4d}步 | {} vs {} | {}".format(
                moves, 
                lg["players"]["player1"]["name"], 
                lg["players"]["player2"]["name"],
                winner
            ))
        print()
    
    # 6. 开局战术分析（基于前 3 步翻牌）
    print("【开局特征分析】")
    print("-" * 70)
    
    first_flip_patterns = Counter()
    for g in games:
        if g.get("error"):
            continue
        opening = g.get("opening_summary", [])
        if len(opening) >= 2:
            player1_first = opening[0].get("piece", "")
            player2_first = opening[1].get("piece", "")
            pattern = "{} vs {}".format(player1_first or "unknown", player2_first or "unknown")
            first_flip_patterns[pattern] += 1
    
    print("常见首手翻牌组合:")
    for pattern, count in first_flip_patterns.most_common(10):
        pct = 100*count/len(games)
        print("  {}: {:5d}次 ({:.1f}%)".format(pattern, count, pct))
    print()
    
    # 7. 玩家对战记录
    print("【热门玩家对战】")
    print("-" * 70)
    
    player_matches = defaultdict(int)
    for g in games:
        if g.get("error"):
            continue
        p1 = g["players"]["player1"]["name"]
        p2 = g["players"]["player2"]["name"]
        player_matches[(p1, p2)] += 1
    
    sorted_matches = sorted(player_matches.items(), key=lambda x: -x[1])[:10]
    if sorted_matches:
        print("出现最多的对战组合:")
        for (p1, p2), count in sorted_matches:
            print("  - {} vs {}: {}局".format(p1, p2, count))
    print()
    
    # 8. 导出 JSON 明细
    print("【数据导出】")
    print("-" * 70)
    output_files = []
    
    # 导出完整数据
    with open('replay_analysis_full.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    output_files.append('replay_analysis_full.json (完整数据)')
    
    # 只导出已结束的胜负局
    decided_games = [g for g in games if not g.get("error") and g["result"]["winner"] in (0, 1)]
    result_only = {
        "metadata": {"description": "Only decisive games", "count": len(decided_games)},
        "games": decided_games
    }
    with open('replay_decisive_games.json', 'w', encoding='utf-8') as f:
        json.dump(result_only, f, ensure_ascii=False, indent=2)
    output_files.append('replay_decisive_games.json (仅胜负局 {})'.format(len(decided_games)))
    
    # 导出已完成但无结果的游戏（可能是中途退出）
    undecided_games = [g for g in games if not g.get("error") and g["result"]["winner"] is None]
    undecided_data = {
        "metadata": {"description": "Undecided games (incomplete)", "count": len(undecided_games)},
        "games": undecided_games
    }
    with open('replay_undecided_games.json', 'w', encoding='utf-8') as f:
        json.dump(undecided_data, f, ensure_ascii=False, indent=2)
    output_files.append('replay_undecided_games.json (未完成局 {})'.format(len(undecided_games)))
    
    for outfile in output_files:
        print("  [OK] 已导出:", outfile)
    
    print()
    print("=" * 70)
    print("报告生成完成!")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='分析军棋复盘数据')
    parser.add_argument('input_file', default='all_replays.json', help='输入 JSON 文件')
    
    args = parser.parse_args()
    analyze_replays(args.input_file)
