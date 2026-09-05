#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""军棋复盘数据查询工具"""

import json
import sys
from collections import Counter

def load_replays(json_file='all_replays.json'):
    """加载复盘数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_by_player(games, player_name):
    """过滤特定玩家的对局"""
    return [g for g in games if 
            player_name in g['players']['player1']['name'] or 
            player_name in g['players']['player2']['name']]

def filter_decided_games(games):
    """只返回已结束的对局"""
    return [g for g in games if g['result']['winner'] is not None]

def get_player_stats(games, player_name):
    """计算某玩家的胜率统计"""
    filtered = [g for g in games if 'players' in g and (
            player_name in g['players']['player1']['name'] or 
            player_name in g['players']['player2']['name'])]
    
    total = len(filtered)
    wins = 0
    losses = 0
    
    for g in filtered:
        winner_id = g['result']['winner']
        p1_name = g['players']['player1']['name']
        p2_name = g['players']['player2']['name']
        
        if winner_id == 0 and p1_name == player_name:
            wins += 1
        elif winner_id == 1 and p2_name == player_name:
            wins += 1
        elif winner_id != -1:  # 非和棋则算输
            losses += 1
    
    print('\n玩家 {} 统计:'.format(player_name))
    print('  总对局数：{}'.format(total))
    print('  胜场：{} ({:.1f}%)'.format(wins, 100*wins/total) if total > 0 else '  胜场：N/A')
    print('  负场：{} ({:.1f}%)'.format(losses, 100*losses/total) if total > 0 else '  负场：N/A')
    
    return {'total': total, 'wins': wins, 'losses': losses}

def analyze_opener_patterns(games):
    """分析开局翻牌模式"""
    patterns = Counter()
    
    for g in games:
        opening = g.get('opening_summary', [])
        if len(opening) >= 2:
            flip1 = opening[0].get('piece', 'unknown')
            flip2 = opening[1].get('piece', 'unknown')
            patterns[(flip1, flip2)] += 1
    
    print("\n常见开局翻牌组合（前两步）:")
    for (p1, p2), count in patterns.most_common(15):
        pct = 100 * count / len(games)
        print(f"  '{p1}' vs '{p2}': {count}局 ({'%.1f' % pct}%)")

def show_game_detail(games, filename):
    """显示某局对局的详细信息"""
    for g in games:
        if g['filename'] == filename:
            print(f"\n=== 对局详情：{filename} ===")
            print(f"玩家 1: {g['players']['player1']['name']} (等级分：{g['players']['player1']['rating']})")
            print(f"玩家 2: {g['players']['player2']['name']} (等级分：{g['players']['player2']['rating']})")
            print(f"对局模式：{g['game_info']['mode']}")
            print(f"总着法数：{g['game_info']['total_moves']}")
            print(f"回放步数：{g['game_info']['replay_plies']}")
            print(f"胜负结果：{g['result']['win_reason']}")
            print(f"回放成功：{g['result']['replay_ok']}")
            
            if g.get('opening_summary'):
                print("\n开局前 3 步:")
                for step in g['opening_summary'][:3]:
                    print(f"  第{step['step']}步：{step}")
            return
    
    print(f"[警告] 未找到文件：{filename}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='查询军棋复盘数据')
    parser.add_argument('json_file', nargs='?', default='all_replays.json', help='输入 JSON 文件')
    parser.add_argument('--by-player', '-p', metavar='PLAYER', help='按玩家名查询')
    parser.add_argument('--stats', '-s', metavar='PLAYER', help='查看某玩家统计')
    parser.add_argument('--openers', '-o', action='store_true', help='分析开局模式')
    parser.add_argument('--detail', '-d', metavar='FILENAME', help='显示某局详情')
    parser.add_argument('--decided', '-D', action='store_true', help='只显示已结束对局')
    
    args = parser.parse_args()
    
    print(f'正在加载 {args.json_file}...')
    data = load_replays(args.json_file)
    games = data['games']
    
    if args.decided:
        games = filter_decided_games(games)
        print(f'[OK] 筛选后剩余 {len(games)} 局已结束对局\n')
    
    if args.by_player:
        filtered = filter_by_player(data['games'], args.by_player)
        print(f'\n包含 "{args.by_player}" 的对局共 {len(filtered)} 局:')
        for g in filtered[:10]:
            moves = g['game_info']['total_moves']
            winner = g['result']['winner']
            winner_str = ['先手胜', '后手胜'][winner] if winner in [0,1] else ['和棋', '未终局'][winner]
            print(f"  - {g['filename']}: {moves}步 | {winner_str}")
        if len(filtered) > 10:
            print(f'  ... 还有 {len(filtered)-10} 局')
    
    if args.stats:
        get_player_stats(data['games'], args.stats)
    
    if args.openers:
        analyze_opener_patterns(data['games'])
    
    if args.detail:
        show_game_detail(data['games'], args.detail)
    
    # 默认显示基础统计
    if not any([args.by_player, args.stats, args.openers, args.detail, args.decided]):
        stats = data['statistics']
        decided = stats['winners']['player1'] + stats['winners']['player2']
        success_pct = 100 * stats['valid_games'] / stats['total_games']
        print('\n快速统计:')
        print('  总复盘数：{}'.format(stats['total_games']))
        print('  成功解析：{} ({:.1f}%)'.format(stats['valid_games'], success_pct))
        print('  已结束：{}局 | 未完成：{}局 | 和棋：{}局'.format(
            decided, stats['winners']['undecided'], stats['winners']['draw']))
        print('  平均着法数：{:.1f}步'.format(stats['move_stats']['avg']))

if __name__ == '__main__':
    main()
