"""统一复盘会话管理器 (Replay Manager)。

支持解析与步进重放：
1. 原生 App .sav 二进制复盘文件 (parse_sav / replay.py)；
2. 人机对战 JSON 记录文件 (games/game_*.json)。

对外提供统一的 ReplaySession 抽象，包含每步盘面快照 (GameState)、
执行动作 (Action)、动作解析描述 (action_desc) 以及 AI 评分元数据。
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import RuleConfig
from .replay import SPECIAL_EVENT, cell_rc, parse_sav, board_from_table
from .rules import COLOR_CN, RANK_CN, Rank
from .state import Action, GameState, Piece, deal


def describe_action(st_before: GameState, action: Action, st_after: GameState) -> str:
    """生成易于人类阅读的着法描述（如：红连长 (5,2)->(5,3) 吃 蓝排长）。"""
    if action.kind == "flip":
        p = st_after.board.get(action.frm)
        if p and p.revealed:
            clr = COLOR_CN.get(p.color, p.color)
            rnk = RANK_CN.get(p.rank, str(p.rank))
            return f"翻开 ({action.frm[0]},{action.frm[1]}) 见 【{clr}{rnk}】"
        return f"翻开 ({action.frm[0]},{action.frm[1]})"

    # move
    p_from = st_before.board.get(action.frm)
    p_to = st_before.board.get(action.to)
    from_str = f"({action.frm[0]},{action.frm[1]})"
    to_str = f"({action.to[0]},{action.to[1]})"

    if not p_from:
        return f"走 {from_str}->{to_str}"

    c_from = COLOR_CN.get(p_from.color, p_from.color)
    r_from = RANK_CN.get(p_from.rank, str(p_from.rank))
    mover_str = f"{c_from}{r_from}"

    if not p_to:
        return f"{mover_str} {from_str}->{to_str}"

    c_to = COLOR_CN.get(p_to.color, p_to.color)
    r_to = RANK_CN.get(p_to.rank, str(p_to.rank))
    target_str = f"{c_to}{r_to}"

    # 检查移动后目标格棋子情况判断是吃还是同归于尽
    after_tgt = st_after.board.get(action.to)
    if after_tgt is not None and after_tgt.color == p_from.color:
        return f"{mover_str} {from_str}->{to_str} 吃 {target_str}"
    else:
        return f"{mover_str} {from_str}->{to_str} 同尽 {target_str}"


@dataclass
class ReplayStep:
    """单步复盘记录快照。"""
    ply: int
    seat: int
    turn_color: Optional[str]
    action: Action
    action_desc: str
    state_before: GameState
    state_after: GameState
    ai_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplaySession:
    """统一复盘会话对象。"""
    file_path: str
    game_type: str  # "sav" | "json"
    metadata: Dict[str, Any]
    initial_state: GameState
    steps: List[ReplayStep] = field(default_factory=list)

    @property
    def total_plies(self) -> int:
        return len(self.steps)

    @property
    def final_state(self) -> GameState:
        if self.steps:
            return self.steps[-1].state_after
        return self.initial_state

    def get_step(self, ply: int) -> Optional[ReplayStep]:
        if 0 <= ply < len(self.steps):
            return self.steps[ply]
        return None

    def get_state_at(self, ply: int) -> GameState:
        """获取执行完第 ply 手之后的状态；ply=-1 获取初始状态。"""
        if ply < 0 or not self.steps:
            return self.initial_state
        if ply >= len(self.steps):
            return self.steps[-1].state_after
        return self.steps[ply].state_after


class ReplayManager:
    """复盘文件加载与会话构建器。"""

    @staticmethod
    def load_game(file_path: str, cfg: Optional[RuleConfig] = None) -> ReplaySession:
        """根据文件扩展名自动分发并加载复盘。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"复盘文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".sav":
            return ReplayManager.load_sav(file_path, cfg=cfg)
        elif ext == ".json":
            return ReplayManager.load_json(file_path, cfg=cfg)
        else:
            raise ValueError(f"不支持的复盘文件格式: {ext} (仅支持 .sav 与 .json)")

    @staticmethod
    def load_sav(file_path: str, cfg: Optional[RuleConfig] = None) -> ReplaySession:
        """解析原生 .sav 二进制复盘文件并生成逐步快照。"""
        sav_game = parse_sav(file_path)
        # 为避免历史棋谱回放中因长局 quiet 达到 40 步被提前截断，回放时采用 70 步时钟
        replay_cfg = cfg or RuleConfig(no_capture_draw_plies=70)

        board = board_from_table(sav_game.table)
        st = GameState(board=board, cfg=replay_cfg)
        initial_state = st.copy()

        steps: List[ReplayStep] = []
        for i, (a, b, c) in enumerate(sav_game.moves):
            if (a, b, c) == SPECIAL_EVENT:
                break
            if a == b and c == 1:
                act = Action("flip", cell_rc(a))
            elif a != b and c in (1, 3):
                act = Action("move", cell_rc(a), cell_rc(b))
            else:
                break

            st_before = st
            turn_color = st.my_color()
            seat = st.turn

            st_after = st.apply(act)
            desc = describe_action(st_before, act, st_after)

            steps.append(ReplayStep(
                ply=i,
                seat=seat,
                turn_color=turn_color,
                action=act,
                action_desc=desc,
                state_before=st_before,
                state_after=st_after,
                ai_meta={},
            ))
            st = st_after
            if st.is_terminal():
                break

        metadata = {
            "source": os.path.basename(file_path),
            "version": sav_game.version,
            "names": sav_game.names,
            "ratings": sav_game.ratings,
            "mode": sav_game.mode,
            "ai_level": sav_game.ai_level,
            "timestamp": sav_game.timestamp,
            "winner": st.winner,
            "win_reason": st.win_reason,
            "total_plies": len(steps),
        }

        return ReplaySession(
            file_path=file_path,
            game_type="sav",
            metadata=metadata,
            initial_state=initial_state,
            steps=steps,
        )

    @staticmethod
    def load_json(file_path: str, cfg: Optional[RuleConfig] = None) -> ReplaySession:
        """解析人机对战 JSON 记录文件 (games/*.json) 并生成逐步快照。"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        seed = data.get("seed", 0)
        # 对战复盘回放使用 70 步容错避免提前截断
        replay_cfg = cfg or RuleConfig(no_capture_draw_plies=70)
        st = deal(random.Random(seed), replay_cfg)
        initial_state = st.copy()

        steps: List[ReplayStep] = []
        moves_data = data.get("moves", [])
        for m in moves_data:
            kind = m.get("kind")
            frm = tuple(m["frm"])
            to = tuple(m["to"]) if m.get("to") is not None else None
            act = Action(kind, frm, to)

            st_before = st
            turn_color = st.my_color()
            seat = m.get("seat", st.turn)

            ai_meta = {
                "engine": m.get("engine", data.get("engine_type")),
                "ai_seed": m.get("ai_seed"),
                "top_scored": m.get("top_scored", []),
            }

            st_after = st.apply(act)
            desc = describe_action(st_before, act, st_after)

            steps.append(ReplayStep(
                ply=len(steps),
                seat=seat,
                turn_color=turn_color,
                action=act,
                action_desc=desc,
                state_before=st_before,
                state_after=st_after,
                ai_meta=ai_meta,
            ))
            st = st_after
            if st.is_terminal():
                break

        metadata = {
            "source": os.path.basename(file_path),
            "seed": seed,
            "human_seat": data.get("human_seat", 0),
            "engine_type": data.get("engine_type", "unknown"),
            "depth": data.get("depth", 3),
            "winner": data.get("winner", st.winner),
            "win_reason": data.get("reason", st.win_reason),
            "total_plies": len(steps),
        }

        return ReplaySession(
            file_path=file_path,
            game_type="json",
            metadata=metadata,
            initial_state=initial_state,
            steps=steps,
        )
