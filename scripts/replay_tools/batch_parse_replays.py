#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量解析军棋复盘文件并导出为 JSON"""

import sys
sys.path.insert(0, '.')

import os
import json
from pathlib import Path
from junqi import replay
from junqi.rules import COLOR_CN, RANK_CN
from datetime import datetime

def parse_all_replays(input_dir: str, output_file: str):
    """批量解析 .sav 文件并导出为结构化 JSON"""
    
    # 查找所有 .sav 文件
    if os.path.isfile(input_dir):
        sav_files = [input_dir]
    else:
        sav_files = sorted([
            os.path.join(input_dir, f) 
            for f in os.listdir(input_dir) 
            if f.lower().endswith('.sav')
        ])
    
    print(f'找到 {len(sav_files)} 个复盘文件...')
    
    results = {
        "metadata": {
            "total_files": len(sav_files),
            "parse_time": datetime.now().isoformat(),
            "source_directory": input_dir
        },
        "games": [],
        "statistics": {}
    }
    
    successful = 0
    failed = 0
    
    for i, filepath in enumerate(sav_files):
        filename = os.path.basename(filepath)
        
        try:
            # 解析单个文件
            game = replay.parse_sav(filepath)
            replay_result = replay.replay_sav(game)
            
            # 提取关键信息
            game_data = {
                "file_index": i + 1,
                "filename": filename,
                "players": {
                    "player1": {"name": game.names[0], "rating": game.ratings[0]},
                    "player2": {"name": game.names[1], "rating": game.ratings[1]}
                },
                "game_info": {
                    "mode": game.mode,
                    "ai_level": game.ai_level,
                    "total_moves": game.n_moves,
                    "replay_plies": replay_result.n_plies,
                    "timestamp_approx": game.timestamp
                },
                "result": {
                    "winner": replay_result.winner,
                    "win_reason": replay_result.win_reason,
                    "stopped_on_event": replay_result.stopped_on_event,
                    "replay_ok": replay_result.replay_ok,
                    "error_message": replay_result.error
                },
                "opening_summary": extract_opening_features(replay_result.final_state, game.moves)
            }
            
            results["games"].append(game_data)
            successful += 1
            
            # 进度显示
            if (i + 1) % 100 == 0:
                print('[OK] 已处理 {}/{} ({})'.format(i+1, len(sav_files), successful))
                
        except Exception as e:
            failed += 1
            print('[X] 解析失败 {}: {}'.format(filename, e))
            results["games"].append({
                "file_index": i + 1,
                "filename": filename,
                "error": str(e)
            })
    
    # 计算统计信息
    results["statistics"] = calculate_statistics(results["games"])
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print('\n[SUCCESS] 批量解析完成!')
    print('  成功：{}/{}'.format(successful, len(sav_files)))
    print('  失败：{}/{}'.format(failed, len(sav_files)))
    print('  输出文件：{}'.format(output_file))
    
    return results

def extract_opening_features(final_state, moves):
    """提取开局特征（前 6 步）"""
    opening = []
    for i, (a, b, c) in enumerate(moves[:6]):
        from_cell = replay.cell_rc(a)
        to_cell = replay.cell_rc(b)
        piece_type = ""
        
        if a == b and c == 1:  # 翻牌
            piece = final_state.board.get(from_cell) if i < len(moves) else None
            if piece:
                piece_type = f"{COLOR_CN.get(piece.color, '')}{RANK_CN.get(piece.rank, '')}"
            opening.append({
                "step": i + 1,
                "type": "flip",
                "cell": list(from_cell),
                "piece": piece_type
            })
        elif a != b and c in (1, 3):
            opening.append({
                "step": i + 1,
                "type": "move",
                "from": list(from_cell),
                "to": list(to_cell),
                "flag_capture": c == 1 and final_state.board.get(to_cell) is not None,
                "flag_suicide": c == 3
            })
    
    return opening

def calculate_statistics(games):
    """计算对局统计数据"""
    total = len(games)
    
    # 胜负分布
    winners = {"player1": 0, "player2": 0, "draw": 0, "undecided": 0}
    for g in games:
        if g.get("error"):
            continue
        w = g["result"]["winner"]
        if w == 0:
            winners["player1"] += 1
        elif w == 1:
            winners["player2"] += 1
        elif w == -1:
            winners["draw"] += 1
        else:
            winners["undecided"] += 1
    
    # 着法数统计
    move_counts = [g["game_info"]["total_moves"] for g in games if not g.get("error")]
    
    # 模式统计
    mode_counts = {}
    for g in games:
        if g.get("error"):
            continue
        m = g["game_info"]["mode"]
        mode_counts[m] = mode_counts.get(m, 0) + 1
    
    return {
        "total_games": total,
        "valid_games": len(move_counts),
        "winners": winners,
        "move_stats": {
            "min": min(move_counts) if move_counts else 0,
            "max": max(move_counts) if move_counts else 0,
            "avg": sum(move_counts)/len(move_counts) if move_counts else 0
        },
        "modes": mode_counts
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量解析军棋复盘文件')
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output', default='all_replays.json', help='输出 JSON 文件名')
    
    args = parser.parse_args()
    
    parse_all_replays(args.input, args.output)
