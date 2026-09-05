"""规则核心：棋盘几何、铁路网络、走法生成辅助、战斗结算。

棋盘坐标：(row, col)，row 0 = 上方阵地底线（大本营行），row 11 = 下方阵地底线。
row 0-5 为上半场，row 6-11 为下半场，两军前线在 row 5 / row 6 之间。

铁路网（对照实物棋盘）：
  · 横向铁路：row 1、5、6、10（每方前线排 + 每方倒数第二排）
  · 纵向铁路：col 0、4，贯通 row 0-11 全场
  · 四个角点 (1,0)(1,4)(10,0)(10,4) 为弧形弯道，允许横转纵
  · T 形交点 (5,0)(5,4)(6,0)(6,4) 不允许铁路转弯，只能直行
  · 两军前线之间：col 0/4 铁路直通，col 2 为公路单步，col 1/3 不连通
"""
from __future__ import annotations

from enum import IntEnum
from itertools import product

ROWS, COLS = 12, 5


class Rank(IntEnum):
    """棋子等级。SI..GONG 数值越大战斗力越强；ZHA/LEI/QI 为特殊子，
    不参与数值强弱比较，值必须唯一（否则 == 判定会串）。"""

    GONG = 5   # 工兵
    PAI = 6    # 排长
    LIAN = 7   # 连长
    YING = 8   # 营长
    TUAN = 9   # 团长
    LV = 10    # 旅长
    SHI = 11   # 师长
    JUN = 12   # 军长
    SI = 13    # 司令
    ZHA = 20   # 炸弹
    LEI = 21   # 地雷
    QI = 22    # 军旗

    @property
    def is_regular(self) -> bool:
        return 5 <= self.value <= 13


RANK_CN = {
    Rank.SI: "司令", Rank.JUN: "军长", Rank.SHI: "师长", Rank.LV: "旅长",
    Rank.TUAN: "团长", Rank.YING: "营长", Rank.LIAN: "连长", Rank.PAI: "排长",
    Rank.GONG: "工兵", Rank.ZHA: "炸弹", Rank.LEI: "地雷", Rank.QI: "军旗",
}

# 每方 25 枚的构成
COMPOSITION = {
    Rank.SI: 1, Rank.JUN: 1, Rank.SHI: 2, Rank.LV: 2, Rank.TUAN: 2,
    Rank.YING: 2, Rank.LIAN: 3, Rank.PAI: 3, Rank.GONG: 3,
    Rank.ZHA: 2, Rank.LEI: 3, Rank.QI: 1,
}

COLORS = ("r", "b")  # 棋子印刷颜色：红(r)/蓝(b)，首翻定色后归属双方
COLOR_CN = {"r": "红", "b": "蓝"}


def other(color: str) -> str:
    return "b" if color == "r" else "r"


# ---------------------------------------------------------------- 棋盘几何

def in_board(p) -> bool:
    return 0 <= p[0] < ROWS and 0 <= p[1] < COLS


CAMPS = frozenset({  # 行营（安全区，开局不放子）
    (2, 1), (2, 3), (3, 2), (4, 1), (4, 3),      # 上方
    (7, 1), (7, 3), (8, 2), (9, 1), (9, 3),      # 下方
})

HQS = frozenset({(0, 1), (0, 3), (11, 1), (11, 3)})  # 大本营：进入后不能再动

RAIL_ROWS = frozenset({1, 5, 6, 10})
RAIL_COLS = frozenset({0, 4})

# 铁路网：横向 row 1, 5, 6, 10，纵向 col 0, 4 仅在 row 1..10（两端不到达底线 row 0 与 row 11）
RAIL_POSITIONS = frozenset(
    {(r, c) for r in RAIL_ROWS for c in range(COLS)} |
    {(r, c) for r in range(1, 11) for c in RAIL_COLS}
)


def is_rail(p) -> bool:
    return p in RAIL_POSITIONS


def is_camp(p) -> bool:
    return p in CAMPS


def is_hq(p) -> bool:
    return p in HQS


PLAY_POSITIONS = tuple(  # 开局 50 个放子位置（非行营），按行列稳定排序
    (r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in CAMPS
)
ALL_POSITIONS = tuple((r, c) for r in range(ROWS) for c in range(COLS))

# 铁路角点与 T 形交点：仅用于统计分桶。App 实测铁路一律走直线、不拐弯
# （见 config.RuleConfig），走法层不再做任何转弯。
CORNER_JUNCTIONS = frozenset({(1, 0), (1, 4), (10, 0), (10, 4)})
T_JUNCTIONS = frozenset({(5, 0), (5, 4), (6, 0), (6, 4)})

# 前线三通道的铁路阻断：col1/col3 完全不通（公路层已断开），
# col2 是公路（可一步走过，但铁路滑行不可穿越）。
CROSS_BLOCKED = frozenset({
    frozenset(((5, 1), (6, 1))),
    frozenset(((5, 2), (6, 2))),
    frozenset(((5, 3), (6, 3))),
})

# 工兵绕行禁区：已废弃。早期"App 实测"认为双方底线（横铁路 row 1/10）工兵
# 不转弯，被 1000 局真实复盘数据推翻——255 处底线转弯着法全部合法
# （见 junqi/replay.py 回放校验）。App 实为全场任意转弯的经典规则。
# 保留空集常量以兼容引用。
ENGINEER_NO_TURN_ROWS = frozenset()


def _build_adjacency():
    """公路一步邻接 = 全场正交边 + 行营 4 条斜向人道 − 前线 col1/3 断开。"""
    road = set()
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr < ROWS and nc < COLS:
                    road.add((min((r, c), (nr, nc)), max((r, c), (nr, nc))))
    for (r, c) in CAMPS:  # 行营斜向人道
        for dr, dc in product((-1, 1), (-1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                road.add((min((r, c), (nr, nc)), max((r, c), (nr, nc))))
    road.discard(((5, 1), (6, 1)))  # 前线只有 col 0/2/4 连通
    road.discard(((5, 3), (6, 3)))
    neighbors = {p: [] for p in ALL_POSITIONS}
    for a, b in road:
        neighbors[a].append(b)
        neighbors[b].append(a)
    return neighbors


NEIGHBORS = _build_adjacency()

# 铁路正交邻接（工兵飞行用；前线三通道按 CROSS_BLOCKED 阻断）
_RAIL_STEP = {
    p: [n for n in NEIGHBORS[p]
        if abs(n[0] - p[0]) + abs(n[1] - p[1]) == 1 and is_rail(n)
        and frozenset((p, n)) not in CROSS_BLOCKED]
    for p in ALL_POSITIONS if is_rail(p)
}

DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def rail_neighbors(p):
    return _RAIL_STEP.get(p, ())


# ---------------------------------------------------------------- 战斗结算

ATTACKER_WINS = "attacker_wins"
DEFENDER_WINS = "defender_wins"
BOTH_DIE = "both_die"


def battle(attacker: Rank, defender: Rank) -> str:
    """返回战斗结果。调用方保证 attacker 不是 LEI/QI，且军旗攻击合法性已在走法层把关。"""
    if defender == Rank.QI:
        return ATTACKER_WINS          # 扛旗成功（炸弹同尽见下）
    if attacker == Rank.ZHA or defender == Rank.ZHA:
        return BOTH_DIE
    if defender == Rank.LEI:
        return ATTACKER_WINS if attacker == Rank.GONG else DEFENDER_WINS
    if attacker == defender:
        return BOTH_DIE
    return ATTACKER_WINS if attacker > defender else DEFENDER_WINS
