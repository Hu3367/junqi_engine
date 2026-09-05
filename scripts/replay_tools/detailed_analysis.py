#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复盘详细分析脚本"""

import sys
sys.path.insert(0, '.')

from junqi import replay
from junqi.rules import COLOR_CN, RANK_CN

# 加载复盘
g = replay.parse_sav('军旗复盘/251122223849 棋手 62240-棋手 44374.sav')
r = replay.replay_sav(g)

# 打印基础信息
print('=' * 70)
print('军棋复盘分析报告')
print('=' * 70)
print('玩家 1:', g.names[0], '(等级分：', g.ratings[0], ')')
print('玩家 2:', g.names[1], '(等级分：', g.ratings[1], ')')
print('对局模式:', g.mode, '(1=天梯)')
print('总着法数:', g.n_moves)
print('回放步数:', r.n_plies)
print('最终状态:', '未终局' if r.winner is None else r.win_reason)
print()

# 逐手分析
st = replay.GameState(board=replay.board_from_table(g.table))
color_map = {"r": "红方", "b": "蓝方"}

print('逐步分析:')
print('-' * 70)
for i, (a, b, c) in enumerate(g.moves):
    ply = i + 1
    seat = (ply - 1) % 2
    player_name = g.names[seat]
    assigned_color = st.seat_color.get(seat)
    color_str = color_map.get(assigned_color, "") if assigned_color else ""
    
    from_cell, to_cell = replay.cell_rc(a), replay.cell_rc(b)
    
    if a == b and c == 1:  # 翻牌
        piece = st.board.get(from_cell)
        if piece:
            piece_str = COLOR_CN.get(piece.color, '') + RANK_CN.get(piece.rank, '')
            print('第 {:2d} 手 | {:12s} [{}] 翻开 ({}, {}) -> 【{}】'.format(
                i+1, player_name, color_str, from_cell[0], from_cell[1], piece_str))
    elif a != b and c == 1:  # 移动或吃子
        src_p = st.board.get(from_cell)
        dst_p = st.board.get(to_cell)
        if src_p:
            piece_str = COLOR_CN.get(src_p.color, '') + RANK_CN.get(src_p.rank, '')
            if dst_p:
                target_str = COLOR_CN.get(dst_p.color, '') + RANK_CN.get(dst_p.rank, '')
                print('第 {:2d} 手 | {:12s} [{}] 【{}】在 ({}, {}) 吃掉【{}】'.format(
                    i+1, player_name, color_str, piece_str, to_cell[0], to_cell[1], target_str))
            else:
                print('第 {:2d} 手 | {:12s} [{}] 【{}】移动到 ({}, {})'.format(
                    i+1, player_name, color_str, piece_str, to_cell[0], to_cell[1]))
    elif a != b and c == 3:  # 同归于尽
        src_p = st.board.get(from_cell)
        if src_p:
            piece_str = COLOR_CN.get(src_p.color, '') + RANK_CN.get(src_p.rank, '')
            print('第 {:2d} 手 | {:12s} [{}] 【{}】与目标同归于尽'.format(
                i+1, player_name, color_str, piece_str))
    
    # 更新状态
    if a == b:
        step = replay.Action("flip", from_cell)
    else:
        step = replay.Action("move", from_cell, to_cell)
    st = st.apply(step)

print('=' * 70)
