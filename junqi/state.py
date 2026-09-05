"""对局状态：发牌、走法生成、事件应用、隐藏信息概率模型、JSON 记谱。"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from typing import Optional

from .config import RuleConfig
from .rules import (ATTACKER_WINS, BOTH_DIE, CAMPS, COLOR_CN, COLORS, COLS,
                    COMPOSITION, CORNER_JUNCTIONS, CROSS_BLOCKED, DIRS,
                    ENGINEER_NO_TURN_ROWS, HQS, NEIGHBORS, PLAY_POSITIONS,
                    RANK_CN, Rank, battle, in_board, is_camp, is_hq, is_rail,
                    other, rail_neighbors)

WIN_SCORE = 1_000_000


def position_key(st: "GameState"):
    """可观察局面键（双方明子、暗子位置、轮次、阵营归属）。

    APK 判循环和按"相同局面"计——玩家只能看到明子与暗子位置，看不到暗子
    身份，因此键里暗子只记位置不记身份（与 selfplay 旧的全信息键不同）。
    """
    obs = []
    for pos in sorted(st.board):
        pc = st.board[pos]
        obs.append((pos, (pc.color, pc.rank.name) if pc.revealed else None))
    return (tuple(obs), st.turn,
            st.seat_color.get(0), st.seat_color.get(1))


@dataclass(frozen=True)
class Piece:
    color: str
    rank: Rank
    revealed: bool = False

    @property
    def name(self) -> str:
        return f"{self.color}{RANK_CN[self.rank]}"


@dataclass(frozen=True)
class Action:
    kind: str                 # 'flip' | 'move'
    frm: tuple                # flip: 暗子位置；move: 起点
    to: Optional[tuple] = None

    def __str__(self):
        if self.kind == "flip":
            return f"翻({self.frm[0]},{self.frm[1]})"
        return f"走({self.frm[0]},{self.frm[1]})->({self.to[0]},{self.to[1]})"


def deal(rng: Optional[random.Random] = None,
         cfg: Optional[RuleConfig] = None) -> "GameState":
    """洗牌发牌：50 子随机铺满非行营位置，座位 0 先行。"""
    rng = rng or random.Random()
    pieces = [Piece(color, rank) for color in COLORS
              for rank, n in COMPOSITION.items() for _ in range(n)]
    rng.shuffle(pieces)
    board = {pos: pieces[i] for i, pos in enumerate(PLAY_POSITIONS)}
    return GameState(board=board, cfg=cfg or RuleConfig())


class GameState:
    __slots__ = ("board", "dead", "seat_color", "turn", "first_flip_done",
                 "ply", "quiet", "winner", "win_reason", "cfg")

    def __init__(self, board, dead=(), seat_color=None, turn=0,
                 first_flip_done=False, ply=0, winner=None, win_reason=None,
                 cfg=None, quiet=0):
        self.board = board                  # dict[(r,c) -> Piece]
        self.dead = tuple(dead)             # 已阵亡棋子
        self.seat_color = seat_color or {0: None, 1: None}
        self.turn = turn
        self.first_flip_done = first_flip_done
        self.ply = ply
        self.quiet = quiet                  # 连续无吃子手数（APK 和棋规则）
        self.winner = winner                # None / 0 / 1 / -1(和棋)
        self.win_reason = win_reason        # 'flag'|'immobilized'|'no_capture'|'max_plies'|'draw'
        self.cfg = cfg or RuleConfig()

    # ------------------------------------------------------------- 基础查询

    def copy(self) -> "GameState":
        return GameState(dict(self.board), self.dead, dict(self.seat_color),
                         self.turn, self.first_flip_done, self.ply,
                         self.winner, self.win_reason, self.cfg, self.quiet)

    def my_color(self, seat: Optional[int] = None) -> Optional[str]:
        return self.seat_color[self.turn if seat is None else seat]

    def hidden_positions(self):
        return [p for p, pc in self.board.items() if not pc.revealed]

    def is_terminal(self) -> bool:
        return self.winner is not None

    # ------------------------------------------------------------- 隐藏信息

    def remaining_types(self):
        """公开信息下暗子池的精确构成 {(color, rank): 数量}：
        双方总构成 − 已翻开明子 − 阵亡子。对真实发牌局与手动录入局同样成立。"""
        from collections import Counter
        rem = Counter({(c, r): n for c in COLORS
                       for r, n in COMPOSITION.items()})
        for pc in self.board.values():
            if pc.revealed:
                rem[(pc.color, pc.rank)] -= 1
        for pc in self.dead:
            rem[(pc.color, pc.rank)] -= 1
        return Counter({k: v for k, v in rem.items() if v > 0})

    def marginal(self, pos=None):
        """暗子身份边缘分布 {(color, rank): 概率}（均匀洗牌下的精确先验）。"""
        rem = self.remaining_types()
        total = sum(rem.values()) or 1
        return {k: n / total for k, n in rem.items()}

    def sample_world(self, rng: random.Random, reveal: bool = False):
        """按公开信息采样一个完整世界：{pos -> Piece}，精确均匀。
        reveal=True 时棋子为翻开状态（PIMC 树内完全信息搜索用）。"""
        pool = [Piece(c, r, reveal) for (c, r), n in self.remaining_types().items()
                for _ in range(n)]
        hidden = sorted(self.hidden_positions())
        if len(pool) < len(hidden):
            fallback = [Piece("r", Rank.PAI, reveal), Piece("b", Rank.PAI, reveal)]
            while len(pool) < len(hidden):
                pool.append(rng.choice(fallback))
        rng.shuffle(pool)
        return {pos: pool[i] for i, pos in enumerate(hidden)}

    def instantiate(self, world):
        """在采样世界下生成完全信息状态（PIMC 用）。"""
        b = dict(self.board)
        for pos, pc in world.items():
            b[pos] = pc
        return GameState(b, self.dead, dict(self.seat_color), self.turn,
                         self.first_flip_done, self.ply, self.winner,
                         self.win_reason, self.cfg, self.quiet)

    # ------------------------------------------------------------- 走法生成

    def _flag_attackable(self, hidden) -> bool:
        if self.cfg.flag_needs_all_flipped and hidden:
            return False
        if self.cfg.flag_needs_mines_cleared:
            enemy = other(self.my_color())
            dead_mines = sum(1 for pc in self.dead
                             if pc.color == enemy and pc.rank == Rank.LEI)
            if dead_mines < COMPOSITION[Rank.LEI]:
                return False
        return True

    def _attackable(self, attacker: Rank, target: Piece, tpos, hidden, flag_ok) -> bool:
        if not target.revealed:      # 翻棋标准：不可攻击暗子
            return False
        if target.color == self.my_color():
            return False
        if is_camp(tpos):            # 行营内的子不可被攻击
            return False
        if target.rank == Rank.QI:
            if not flag_ok:
                return False
            # APK 规则（ruleflip.txt）：军旗只有工兵能吃
            if self.cfg.flag_gong_only and attacker != Rank.GONG:
                return False
        if not self.cfg.allow_suicide_attack:   # App 实测：禁止小子撞大子
            if battle(attacker, target.rank) == "defender_wins":
                return False
        return True

    def legal_actions(self):
        acts = []
        my = self.my_color()
        hidden = self.hidden_positions()
        for pos in hidden:                                   # 翻子永远可选
            acts.append(Action("flip", pos))
        if my is None:                                       # 首翻前无明子可走
            return acts
        flag_ok = self._flag_attackable(hidden)

        for pos, pc in self.board.items():
            if not pc.revealed or pc.color != my:
                continue
            if pc.rank in (Rank.LEI, Rank.QI):
                continue
            if self.cfg.hq_locks_pieces and is_hq(pos):      # 部分规则：入大本营锁死
                continue

            def attackable(t, tp, att=pc.rank):
                return self._attackable(att, t, tp, hidden, flag_ok)

            for np in NEIGHBORS[pos]:                        # 公路一步（含行营斜道）
                t = self.board.get(np)
                if t is None or attackable(t, np):
                    acts.append(Action("move", pos, np))
            if is_rail(pos):
                if pc.rank == Rank.GONG and self.cfg.engineer_rail_turns:
                    dests = _engineer_flights(self.board, pos, attackable,
                                              can_fly_over_pieces=self.cfg.engineer_can_fly_over_pieces)
                else:
                    dests = _rail_slides(self.board, pos, attackable)
                for np in dests:
                    acts.append(Action("move", pos, np))
        # 去重：同一(起点,终点)可能由公路一步与铁路滑行/工兵飞行重复生成，
        # 等价 Action 若重复进入动作列表会在 MCTS/网络动作头中分裂成两个同效
        # 子节点，撕裂搜索统计（P0 审计实证：吃旗动作被拆到两个实例）。
        return list(dict.fromkeys(acts))

    # ------------------------------------------------------------- 事件应用

    def apply(self, act: Action) -> "GameState":
        board = dict(self.board)
        dead = list(self.dead)
        sc = dict(self.seat_color)
        winner, reason = self.winner, self.win_reason
        ffd = self.first_flip_done
        quiet = self.quiet + 1                     # 翻子/走子每手 +1，吃子清零

        if act.kind == "flip":
            pc = board[act.frm]
            board[act.frm] = replace(pc, revealed=True)
            if not ffd:                                      # 首翻定色
                sc[self.turn] = pc.color
                sc[1 - self.turn] = other(pc.color)
                ffd = True
        else:
            mover = board.pop(act.frm)
            target = board.get(act.to)
            if target is None:
                board[act.to] = mover
            else:
                quiet = 0                                    # 有吃子，重新计数
                res = battle(mover.rank, target.rank)
                if res == ATTACKER_WINS:
                    dead.append(target)
                    board[act.to] = mover
                    if target.rank == Rank.QI:
                        winner, reason = self.turn, "flag"
                elif res == BOTH_DIE:
                    board.pop(act.to, None)
                    dead.append(target)
                    dead.append(mover)
                    if target.rank == Rank.QI:               # 炸弹与旗同尽（仅旧规则可达）
                        winner, reason = self.turn, "flag"
                else:
                    dead.append(mover)

        nxt = GameState(board, dead, sc, 1 - self.turn, ffd,
                        self.ply + 1, winner, reason, self.cfg, quiet)
        if winner is None:
            # APK 和棋规则：连续 70 步未吃子判和；双方总步数达 1000 判和
            if self.cfg.no_capture_draw_plies \
                    and nxt.quiet >= self.cfg.no_capture_draw_plies:
                nxt.winner, nxt.win_reason = -1, "no_capture"
            elif nxt.ply >= self.cfg.max_plies:
                nxt.winner, nxt.win_reason = -1, "max_plies"
            elif not nxt.hidden_positions() and not nxt._has_any_move():
                nxt.winner, nxt.win_reason = self.turn, "immobilized"
        return nxt

    def _has_any_move(self) -> bool:
        """无暗子可翻时，当前行动方是否还有子可动（快速终局判定）。"""
        my = self.my_color()
        if my is None:
            return False
        flag_ok = self._flag_attackable([])
        for pos, pc in self.board.items():
            if not pc.revealed or pc.color != my or pc.rank in (Rank.LEI, Rank.QI):
                continue
            if self.cfg.hq_locks_pieces and is_hq(pos):
                continue
            for np in NEIGHBORS[pos]:
                t = self.board.get(np)
                if t is None:
                    return True
                if self._attackable(pc.rank, t, np, [], flag_ok):
                    return True
            if is_rail(pos):
                if pc.rank == Rank.GONG and self.cfg.engineer_rail_turns:
                    if _engineer_flights(self.board, pos, lambda t, tp: True):
                        return True
                elif _rail_slides(self.board, pos, lambda t, tp: True):
                    return True
        return False

    # ------------------------------------------------------------- 记谱

    def to_json(self) -> str:
        return json.dumps({
            "turn": self.turn,
            "ply": self.ply,
            "quiet": self.quiet,
            "seat_color": {str(k): v for k, v in self.seat_color.items()},
            "first_flip_done": self.first_flip_done,
            "winner": self.winner,
            "win_reason": self.win_reason,
            "board": [{"r": r, "c": c, "color": pc.color, "revealed": pc.revealed,
                       "rank": pc.rank.name}
                      for (r, c), pc in sorted(self.board.items())],
            "dead": [{"color": pc.color, "rank": pc.rank.name} for pc in self.dead],
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str, cfg: Optional[RuleConfig] = None) -> "GameState":
        d = json.loads(text)
        board = {(e["r"], e["c"]): Piece(e["color"], Rank[e["rank"]], e["revealed"])
                 for e in d["board"]}
        dead = [Piece(e["color"], Rank[e["rank"]], True) for e in d["dead"]]
        return cls(board=board, dead=dead,
                   seat_color={int(k): v for k, v in d["seat_color"].items()},
                   turn=d["turn"], first_flip_done=d["first_flip_done"],
                   ply=d["ply"], winner=d["winner"], win_reason=d["win_reason"],
                   cfg=cfg or RuleConfig(), quiet=d.get("quiet", 0))

    # ------------------------------------------------------------- 显示

    def render(self) -> str:
        head = "     " + "".join(f"  {c}   " for c in range(COLS))
        rows = [head, "    " + "+----" * COLS + "+"]
        for r in range(12):
            cells = []
            for c in range(5):
                pc = self.board.get((r, c))
                if (r, c) in CAMPS:
                    cells.append(" ~~~ " if pc is None else f"{pc.color}{RANK_CN[pc.rank][:1]}营")
                elif pc is None:
                    cells.append("  .  ")
                elif not pc.revealed:
                    cells.append(" ??  ")
                else:
                    cells.append(f" {pc.color}{RANK_CN[pc.rank][:1]}  ")
            rows.append(f"  {r:2d} " + "|".join(cells) + "|")
        rows.append("    " + "+----" * COLS + "+")
        color_info = ", ".join(f"座位{s}={COLOR_CN[v] if v else '未定'}"
                               for s, v in self.seat_color.items())
        state = f"第 {self.ply} 手(和棋线 {self.cfg.max_plies})，无吃子 {self.quiet}(线 {self.cfg.no_capture_draw_plies})，轮到座位 {self.turn}"
        if self.winner is not None:
            state = ("和棋" if self.winner == -1
                     else f"座位 {self.winner} 获胜（{self.win_reason}）")
        return "\n".join(rows) + f"\n{state} | {color_info}\n"


# ------------------------------------------------------- 铁路滑行 / 工兵飞行

def _rail_slides(board, start, attackable):
    """铁路滑行（App 实测：一律走直线，不拐弯）：
    沿直线任意远，不可越过任何棋子，前线三通道不可穿越。"""
    from collections import deque
    results = set()
    seen = set()
    queue = deque()
    for d in DIRS:
        queue.append((start, d))
    while queue:
        pos, d = queue.popleft()
        if (pos, d) in seen:
            continue
        seen.add((pos, d))
        dr, dc = DIRS[d]
        np = (pos[0] + dr, pos[1] + dc)
        if not in_board(np) or not is_rail(np) \
                or frozenset((pos, np)) in CROSS_BLOCKED:
            continue
        t = board.get(np)
        if t is not None:
            if attackable(t, np):
                results.add(np)      # 攻击终点，路到此为止
            continue
        results.add(np)
        queue.append((np, d))
    return results


def _engineer_flights(board, start, attackable, can_fly_over_pieces: bool = False):
    """工兵铁路飞行：
    在铁路网内可任意转弯。
    默认 can_fly_over_pieces=False（标准军棋规则）：不可越过任何棋子；
    当 can_fly_over_pieces=True（兼容旧 APK 变体）：可无视路径棋子阻挡。"""
    from collections import deque
    results = set()
    seen = set()
    queue = deque()
    for d in DIRS:
        queue.append((start, d))
    while queue:
        pos, d = queue.popleft()
        if (pos, d) in seen:
            continue
        seen.add((pos, d))
        dr, dc = DIRS[d]
        np = (pos[0] + dr, pos[1] + dc)
        if not in_board(np) or not is_rail(np) \
                or frozenset((pos, np)) in CROSS_BLOCKED:
            continue
        t = board.get(np)
        if t is not None:
            if attackable(t, np):
                results.add(np)      # 可攻击的终点
            if not can_fly_over_pieces:
                continue             # 被棋子阻挡，不可穿透越子继续前行
        else:
            results.add(np)
        queue.append((np, d))
        for nd in DIRS:
            if nd != d:
                queue.append((np, nd))
    return results
