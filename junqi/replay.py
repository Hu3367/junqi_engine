"""军棋 App 对局记录（.sav）解码与引擎回放。

格式（逆向自 libjunqi.so，1000 局真实样本全量验证）：
  99 字节头 + N × 3 字节着法流，文件大小恒为 99 + 3N，明文未加密。
  头：0=版本0x01；1=随机字节；2-4=u24时间戳>>8；5-15/18-28=玩家名(GBK,11B)；
  16-17/29-30=u16天梯分；31,32=标志字节；33-34=u16记录数；35=模式(1/2/3)；
  36=AI思考量；37=随机字节；38=AI等级；39-98=60B 整盘暗子身份表。
  身份表：cell = row*5+col；0=空(行营)；1-12=色A、13-24=色B
  （码 1=司令…9=工兵 10=炸弹 11=地雷 12=军旗）。
  着法 [a,b,c]：a==b,c==1 → 翻开 cell a；a≠b → 从 a 走到 b（c=1 移动/吃子，
  c=3 同归于尽）；(255,255,1) → 无状态变化特殊事件（疑认输/求和，回放截断）。

回放：座位0=文件玩家1，轮流行棋，首翻定色由 GameState.apply 自动处理。
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

from .config import RuleConfig
from .rules import Rank
from .state import Action, GameState, Piece

HEADER_LEN = 99
SPECIAL_EVENT = (255, 255, 1)

SAV_RANK = {1: Rank.SI, 2: Rank.JUN, 3: Rank.SHI, 4: Rank.LV, 5: Rank.TUAN,
            6: Rank.YING, 7: Rank.LIAN, 8: Rank.PAI, 9: Rank.GONG,
            10: Rank.ZHA, 11: Rank.LEI, 12: Rank.QI}

CELL_POS = [(r, c) for r in range(12) for c in range(5)]   # cell = r*5 + c


def cell_rc(cell: int) -> tuple:
    return CELL_POS[cell]


@dataclass
class SavGame:
    path: str = ""
    version: int = 0
    seed1: int = 0
    timestamp: int = 0            # 秒（256s 分辨率，近似）
    names: tuple = ("", "")       # (玩家1, 玩家2)
    ratings: tuple = (0, 0)
    flags: tuple = (0, 0)         # 头 31/32 字节（rest0/rest1，语义未定）
    mode: int = 0
    ai_think: int = 0
    seed2: int = 0
    ai_level: int = 0
    table: tuple = ()             # 60 cells
    moves: tuple = ()             # ((a,b,c), ...)
    # ---- 回放结果（replay_sav 填充）----
    replay_ok: bool | None = None
    error: str | None = None
    winner: int | None = None     # 0=玩家1 1=玩家2 -1=和棋 None=未终局
    win_reason: str | None = None
    n_plies: int = 0
    stopped_on_event: bool = False
    final_state: GameState | None = field(repr=False, default=None)

    @property
    def n_moves(self) -> int:
        return len(self.moves)

    def to_dict(self) -> dict:
        return {"path": os.path.basename(self.path), "names": self.names,
                "ratings": self.ratings, "mode": self.mode,
                "ai_level": self.ai_level, "n_moves": self.n_moves,
                "replay_ok": self.replay_ok, "error": self.error,
                "winner": self.winner, "win_reason": self.win_reason,
                "n_plies": self.n_plies}


def parse_sav(path: str) -> SavGame:
    """解析单个 .sav 文件。头/长度不合法抛 ValueError。"""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < HEADER_LEN:
        raise ValueError(f"文件过短: {len(data)}B")
    g = SavGame(path=path)
    g.version = data[0]
    g.seed1 = data[1]
    g.timestamp = struct.unpack_from("<I", bytes(data[2:5]) + b"\x00")[0] * 256
    g.names = tuple(data[o:o + 11].split(b"\x00")[0].decode("gbk", "replace")
                    for o in (5, 18))
    g.ratings = struct.unpack_from("<HH", data, 16)[0], \
        struct.unpack_from("<H", data, 29)[0]
    g.flags = (data[31], data[32])
    n = struct.unpack_from("<H", data, 33)[0]
    g.mode, g.ai_think, g.seed2, g.ai_level = data[35], data[36], data[37], data[38]
    g.table = tuple(data[39:99])
    if len(data) != HEADER_LEN + 3 * n:
        raise ValueError(f"长度不符: {len(data)} != {HEADER_LEN}+3*{n}")
    g.moves = tuple(struct.unpack_from("<BBB", data, HEADER_LEN + 3 * i)
                    for i in range(n))
    return g


def board_from_table(table) -> dict:
    """身份表 -> 全暗子棋盘（色A='r'、色B='b' 仅作标记，阵营由首翻定色决定）。"""
    board = {}
    for cell, v in enumerate(table):
        if v == 0:
            continue
        if not 1 <= v <= 24:
            raise ValueError(f"cell{cell} 非法棋子码 {v}")
        color = "r" if v <= 12 else "b"
        board[cell_rc(cell)] = Piece(color, SAV_RANK[v if v <= 12 else v - 12])
    return board


def replay_sav(game: SavGame, cfg: RuleConfig | None = None,
               check: bool = True) -> SavGame:
    """按引擎规则回放着法流，填充胜负/合法性结果。

    check=True 时逐步校验着法在 legal_actions 内（作为引擎兼容性回归）。
    特殊事件 (255,255,1) 视为对局中止标记，截断并保留已完成着法。"""
    cfg = cfg or RuleConfig()
    try:
        board = board_from_table(game.table)
    except ValueError as e:
        game.replay_ok, game.error = False, str(e)
        return game
    st = GameState(board=board, cfg=cfg)
    for i, (a, b, c) in enumerate(game.moves):
        if (a, b, c) == SPECIAL_EVENT:
            game.stopped_on_event = True
            break
        if a == b and c == 1:
            act = Action("flip", cell_rc(a))
        elif a != b and c in (1, 3):
            act = Action("move", cell_rc(a), cell_rc(b))
        else:
            game.replay_ok = False
            game.error = f"第{i}条未知着法 ({a},{b},{c})"
            game.final_state = st
            return game
        if check and act not in st.legal_actions():
            game.replay_ok = False
            game.error = f"第{i}条非法着法 {act} (ply={st.ply})"
            game.final_state = st
            return game
        st = st.apply(act)
    game.n_plies = st.ply
    game.replay_ok = True
    game.winner = st.winner
    game.win_reason = st.win_reason
    game.final_state = st
    return game


def load_dir(path: str, check: bool = True) -> list:
    """解析并回放目录/单文件下全部 .sav，返回按文件名排序的 SavGame 列表。"""
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if f.lower().endswith(".sav"))
    return [replay_sav(parse_sav(f), check=check) for f in files]


# ---------------------------------------------------------------- CLI 报告

def summarize(games: list) -> str:
    from collections import Counter
    n = len(games)
    lines = [f"# .sav 复盘回放报告（{n} 局）", ""]
    bad = [g for g in games if g.replay_ok is False]
    ok = [g for g in games if g.replay_ok]
    lines.append(f"- 解析/回放失败：{len(bad)} 局" +
                 (f"（如 {os.path.basename(bad[0].path)}: {bad[0].error}）" if bad else ""))
    if not ok:
        return "\n".join(lines)
    nmoves = sorted(g.n_moves for g in ok)
    plies = sorted(g.n_plies for g in ok)
    pct = lambda q: nmoves[int(q * (len(nmoves) - 1))]
    lines += [f"- 着法数 min/P25/中位/P75/max：{nmoves[0]}/{pct(.25)}/{pct(.5)}/{pct(.75)}/{nmoves[-1]}",
              f"- 回放 ply min/中位/max：{plies[0]}/{plies[len(plies)//2]}/{plies[-1]}"]
    reasons = Counter()
    for g in ok:
        if g.stopped_on_event:
            reasons["特殊事件中止(疑认输/求和)"] += 1
        elif g.winner is None:
            reasons["未终局"] += 1
        elif g.winner == -1:
            reasons[f"和棋({g.win_reason})"] += 1
        else:
            reasons[f"{g.win_reason}"] += 1
    lines += ["", "## 终局方式", "", "| 方式 | 局数 |", "|---|---|"]
    for k, v in reasons.most_common():
        lines.append(f"| {k} | {v} |")
    decided = [g for g in ok if g.winner in (0, 1)]
    if decided:
        w0 = sum(1 for g in decided if g.winner == 0)
        lines += ["", f"## 可判定胜负 {len(decided)} 局：玩家1(文件首位)胜 "
                  f"{w0} ({100*w0/len(decided):.1f}%)", ""]
        by_mode = Counter(g.mode for g in decided)
        lines += ["| 模式 | 局数 |", "|---|---|"] + \
                 [f"| mode{m} | {c} |" for m, c in sorted(by_mode.items())]
    # 用户视角：名为"棋手 62240"之类的玩家1/2 谁是本人由文件名看，这里只按名字统计
    return "\n".join(lines)
