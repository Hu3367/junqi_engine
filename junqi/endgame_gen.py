"""残局生成器（Endgame Generator - 严谨牌池守恒版）。

用于生成具有针对性的 10~15 子中后盘与尾盘对局，供课程学习与分阶段评测使用。
包含死区/堡垒（fortress）结构与非堡垒对抗局面。
保证 50 枚棋子严格守恒。
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional

from .config import RuleConfig
from .rules import CAMPS, COMPOSITION, HQS, PLAY_POSITIONS, Rank, other
from .state import GameState, Piece


def gen_endgame(rng: random.Random, cfg: Optional[RuleConfig] = None, *,
                material_balance: float = 0.0,
                hidden_k: int = 4,
                my_engineers: int = 1,
                opp_engineers: int = 0,
                fortress: bool = False) -> GameState:
    """生成 10~15 子的尾盘局面，保证 50 子严格守恒。
    - seat 0 执红 'r'，seat 1 执蓝 'b'
    - material_balance: [-1.0, 1.0]，正值表示 seat 0 占优
    - fortress: 是否构筑死区堡垒（对方无工兵且军旗被完全封锁）
    """
    cfg = cfg or RuleConfig()
    my_color = "r"
    opp_color = "b"

    # 初始化双方未分配棋子池
    my_pool = Counter(COMPOSITION)
    opp_pool = Counter(COMPOSITION)

    board: Dict[tuple[int, int], Piece] = {}

    def take_piece(color: str, rank: Rank, revealed: bool = True) -> Piece:
        pool = my_pool if color == my_color else opp_pool
        if pool[rank] > 0:
            pool[rank] -= 1
            return Piece(color, rank, revealed=revealed)
        # 若该等级已取完，退回取兵或连长
        for alt_rk in [Rank.PAI, Rank.LIAN, Rank.YING]:
            if pool[alt_rk] > 0:
                pool[alt_rk] -= 1
                return Piece(color, alt_rk, revealed=revealed)
        # 取任意剩余棋子
        for alt_rk, cnt in list(pool.items()):
            if cnt > 0:
                pool[alt_rk] -= 1
                return Piece(color, alt_rk, revealed=revealed)
        return Piece(color, Rank.PAI, revealed=revealed)

    # 1. 放置己方明军旗
    flag_pos = (11, 1) if rng.random() < 0.5 else (11, 3)
    board[flag_pos] = take_piece(my_color, Rank.QI, revealed=True)

    # 2. 放置地雷与死区
    if fortress:
        opp_engineers = 0
        if flag_pos == (11, 1):
            board[(10, 1)] = take_piece(opp_color, Rank.LEI, revealed=True)
            board[(11, 0)] = take_piece(my_color, Rank.LEI, revealed=True)
            board[(11, 2)] = take_piece(my_color, Rank.LEI, revealed=True)
        else:
            board[(10, 3)] = take_piece(opp_color, Rank.LEI, revealed=True)
            board[(11, 2)] = take_piece(my_color, Rank.LEI, revealed=True)
            board[(11, 4)] = take_piece(my_color, Rank.LEI, revealed=True)
    else:
        board[(10, flag_pos[1])] = take_piece(my_color, Rank.LEI, revealed=True)

    # 3. 放置工兵
    available_pos = [p for p in PLAY_POSITIONS if p not in board]
    rng.shuffle(available_pos)

    for _ in range(my_engineers):
        if available_pos and my_pool[Rank.GONG] > 0:
            board[available_pos.pop()] = take_piece(my_color, Rank.GONG, revealed=True)

    for _ in range(opp_engineers):
        if available_pos and opp_pool[Rank.GONG] > 0:
            board[available_pos.pop()] = take_piece(opp_color, Rank.GONG, revealed=True)

    # 4. 放置行营大子
    camp_list = list(CAMPS)
    rng.shuffle(camp_list)
    if camp_list and available_pos:
        cp = camp_list.pop()
        if cp in available_pos:
            available_pos.remove(cp)
            board[cp] = take_piece(my_color, Rank.SHI, revealed=True)
    if camp_list and available_pos:
        cp = camp_list.pop()
        if cp in available_pos:
            available_pos.remove(cp)
            board[cp] = take_piece(opp_color, Rank.JUN, revealed=True)

    # 5. 补充双方普通战斗子力
    extra_ranks = [Rank.LV, Rank.TUAN, Rank.YING, Rank.LIAN, Rank.PAI]
    if material_balance >= 0:
        my_ranks = [Rank.SI, Rank.JUN] + extra_ranks[:2]
        opp_ranks = extra_ranks[1:4]
    else:
        my_ranks = extra_ranks[2:4]
        opp_ranks = [Rank.SI, Rank.JUN, Rank.SHI]

    for rk in my_ranks:
        if available_pos and len(board) < 14:
            board[available_pos.pop()] = take_piece(my_color, rk, revealed=True)

    for rk in opp_ranks:
        if available_pos and len(board) < 14:
            board[available_pos.pop()] = take_piece(opp_color, rk, revealed=True)

    # 6. 从未分配池中抽取暗子
    for _ in range(min(hidden_k, len(available_pos))):
        c = rng.choice([my_color, opp_color])
        pool = my_pool if c == my_color else opp_pool
        # 挑选池中现有的任意暗子等级
        avail_ranks = [r for r, cnt in pool.items() if cnt > 0]
        if avail_ranks:
            rk = rng.choice(avail_ranks)
            board[available_pos.pop()] = take_piece(c, rk, revealed=False)

    # 7. 剩余池中所有棋子放入 dead，保证 50 子完全闭环
    dead = []
    for rk, cnt in my_pool.items():
        for _ in range(cnt):
            dead.append(Piece(my_color, rk, revealed=True))
    for rk, cnt in opp_pool.items():
        for _ in range(cnt):
            dead.append(Piece(opp_color, rk, revealed=True))

    st = GameState(
        board=board,
        dead=dead,
        seat_color={0: my_color, 1: opp_color},
        turn=0,
        first_flip_done=True,
        ply=rng.randint(30, 80),
        cfg=cfg,
        quiet=rng.randint(0, 35),
    )
    return st
