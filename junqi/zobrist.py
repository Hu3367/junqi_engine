"""军棋 Zobrist 64 位哈希：用于置换表、重复局面检测与状态快速签名。

包含：
- 棋盘 60 格 × (颜色, 军衔, 是否明子) 的独立伪随机数
- 行动方轮次 (Turn)
- 暗子池构成分布指纹
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .rules import COLORS, COMPOSITION, ROWS, COLS, Rank

if TYPE_CHECKING:
    from .state import GameState, Piece


# 使用确定性种子生成 64 位无符号随机数表
_RNG = random.Random(0x4A756E51695A6F62)  # "JunQiZob" ASCII seed
_MASK64 = 0xFFFFFFFFFFFFFFFF


def _rand64() -> int:
    return _RNG.getrandbits(64) & _MASK64


# 1. 棋盘每个位置与棋子状态的 Zobrist 表
# KEY: (pos, color, rank, revealed) -> int64
PIECE_KEYS: dict[tuple[tuple[int, int], str, Rank, bool], int] = {}
for r in range(ROWS):
    for c in range(COLS):
        pos = (r, c)
        for clr in COLORS:
            for rk in Rank:
                for rev in (True, False):
                    PIECE_KEYS[(pos, clr, rk, rev)] = _rand64()

# 2. 轮次键
TURN_KEYS: tuple[int, int] = (_rand64(), _rand64())

# 3. 颜色绑定键 (seat 0 执红 / 执蓝 / 未定)
SEAT_COLOR_KEYS: dict[tuple[str | None, str | None], int] = {
    (None, None): _rand64(),
    ("r", "b"): _rand64(),
    ("b", "r"): _rand64(),
}

# 4. 暗子池剩余量签名表: ((color, rank), count) -> int64
POOL_KEYS: dict[tuple[tuple[str, Rank], int], int] = {}
for clr in COLORS:
    for rk, max_cnt in COMPOSITION.items():
        for cnt in range(max_cnt + 1):
            POOL_KEYS[((clr, rk), cnt)] = _rand64()


def compute_zobrist(state: GameState, include_hidden_identity: bool = False) -> int:
    """计算当前局面的 64 位 Zobrist 哈希。

    参数:
        state: 游戏状态
        include_hidden_identity:
            False (默认): 公共视角哈希（暗子仅包含位置与明暗状态，不包含真实身份；
                          配合暗子池总构成签名，适合人机对弈、置换表与 APK 规则重复判断）。
            True: 完全信息哈希（包含每个暗子采样后的真实身份，用于完全信息子树）。
    """
    h = 0

    # 1. 棋盘子力
    for pos, pc in state.board.items():
        if pc.revealed or include_hidden_identity:
            h ^= PIECE_KEYS[(pos, pc.color, pc.rank, pc.revealed)]
        else:
            # 公共视角：暗子统一使用占位伪标记（以 GONG 且 revealed=False 为代表）
            h ^= PIECE_KEYS[(pos, "r", Rank.GONG, False)]

    # 2. 轮次
    h ^= TURN_KEYS[state.turn]

    # 3. 座次定色
    sc_tuple = (state.seat_color.get(0), state.seat_color.get(1))
    h ^= SEAT_COLOR_KEYS.get(sc_tuple, 0)

    # 4. 公开暗子池构成指纹
    rem = state.remaining_types()
    for clr in COLORS:
        for rk, max_cnt in COMPOSITION.items():
            cnt = rem.get((clr, rk), 0)
            h ^= POOL_KEYS[((clr, rk), cnt)]

    return h & _MASK64
