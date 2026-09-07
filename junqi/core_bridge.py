"""C++ 原生核心 (junqi_core) 与 Python 双轨热拔插网桥。

设计原则：
1. 自动探测：动态检测 C++ 原生动态库 `junqi_core` 是否已编译并可用；
2. 透明降级：若 `junqi_core` 未编译，系统自动、无缝降级为纯 Python 引擎，绝不阻断任何上层业务、测试与 GUI；
3. 统一调度：提供统一的 `search_apk_fast` 与状态转换接口，自动激活 C++ 极致算力（50x~100x 加速）。
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Any

from .config import RuleConfig
from .rules import Rank, battle
from .state import Action, GameState, Piece

logger = logging.getLogger(__name__)

try:
    import junqi_core
    HAS_CPP_CORE = True
except ImportError:
    junqi_core = None
    HAS_CPP_CORE = False


def is_cpp_available() -> bool:
    """返回 C++ 原生动态库是否就绪。"""
    return HAS_CPP_CORE


def state_to_cpp(st: GameState) -> Any:
    """将 Python GameState 转换为 C++ JunqiBoard 对象（需要 HAS_CPP_CORE=True）。"""
    if not HAS_CPP_CORE:
        raise RuntimeError("junqi_core C++ extension is not compiled.")

    b = junqi_core.JunqiBoard()
    b.turn = st.turn
    b.ply = st.ply
    b.quiet = st.quiet
    b.winner = st.winner if st.winner is not None else -2
    b.win_reason = st.win_reason or ""

    # 配置同步
    cfg = st.cfg or RuleConfig()
    b.cfg.flag_needs_mines_cleared = cfg.flag_needs_mines_cleared
    b.cfg.flag_needs_all_flipped = cfg.flag_needs_all_flipped
    b.cfg.flag_gong_only = cfg.flag_gong_only
    b.cfg.allow_suicide_attack = cfg.allow_suicide_attack
    b.cfg.hq_locks_pieces = cfg.hq_locks_pieces
    b.cfg.engineer_rail_turns = cfg.engineer_rail_turns
    b.cfg.engineer_can_fly_over_pieces = cfg.engineer_can_fly_over_pieces
    b.cfg.no_capture_draw_plies = cfg.no_capture_draw_plies
    b.cfg.max_plies = cfg.max_plies
    b.cfg.repetition_draw_count = cfg.repetition_draw_count

    b.first_flip_done = st.first_flip_done

    # 阵营与棋子
    def _color_to_cpp(c: Optional[str]):
        if c == "r" or c == "orange": return junqi_core.Color.RED
        if c == "b" or c == "purple": return junqi_core.Color.BLUE
        return junqi_core.Color.NONE

    b.set_seat_color(0, _color_to_cpp(st.seat_color.get(0)))
    b.set_seat_color(1, _color_to_cpp(st.seat_color.get(1)))

    for (r, c), pc in st.board.items():
        rk_val = getattr(junqi_core.Rank, pc.rank.name, junqi_core.Rank.EMPTY)
        clr_val = _color_to_cpp(pc.color)
        b.set_piece(r, c, rk_val, clr_val, pc.revealed)

    for pc in st.dead:
        rk_val = getattr(junqi_core.Rank, pc.rank.name, junqi_core.Rank.EMPTY)
        clr_val = _color_to_cpp(pc.color)
        b.add_dead(rk_val, clr_val, pc.revealed)

    return b


def cpp_action_to_py(act: Any) -> Action:
    """将 C++ Action 转换为 Python Action。"""
    if act.kind == junqi_core.ActionKind.FLIP:
        return Action(kind="flip", frm=act.frm_pos)
    return Action(kind="move", frm=act.frm_pos, to=act.to_pos)


def search_apk_auto(
    state: GameState,
    level: str = "advanced",
    depth: Optional[int] = None,
    time_limit_ms: Optional[int] = None,
    qsearch_depth: Optional[int] = None,
    avoid: Optional[set] = None,
    prefer_cpp: bool = True,
) -> Tuple[Optional[Action], float, dict]:
    """统一 APK 搜索引擎入口：优先调用 C++ 原生引擎，未编译时平滑降级为 Python 引擎。

    返回: (best_action, score, stats_dict)
    """
    if prefer_cpp and HAS_CPP_CORE:
        try:
            cpp_board = state_to_cpp(state)
            engine = junqi_core.ApkSearchEngine(18, 2026)
            best_act, score = engine.search(
                cpp_board,
                level=level,
                depth_override=depth if depth is not None else -1,
                time_limit_ms_override=time_limit_ms if time_limit_ms is not None else -1,
                qsearch_depth_override=qsearch_depth if qsearch_depth is not None else -1,
            )
            c_stats = engine.get_stats()
            stats_dict = {
                "nodes": c_stats.nodes,
                "qnodes": c_stats.qnodes,
                "tt_hits": c_stats.tt_hits,
                "depth_reached": c_stats.depth_reached,
                "time_spent_ms": c_stats.time_spent_ms,
                "is_cpp": True,
            }
            py_act = cpp_action_to_py(best_act)
            return py_act, score, stats_dict
        except Exception as e:
            logger.warning("C++ engine call failed, falling back to Python: %s", e)

    # 纯 Python 引擎平滑降级
    from .apk_engine import ApkSearchEngine
    py_engine = ApkSearchEngine()
    py_act, score, stats = py_engine.search(
        state,
        level=level,
        depth=depth,
        time_limit_ms=time_limit_ms,
        qsearch_depth=qsearch_depth,
        avoid=avoid,
    )
    stats_dict = {
        "nodes": stats.nodes,
        "qnodes": stats.qnodes,
        "tt_hits": stats.tt_hits,
        "depth_reached": stats.depth_reached,
        "time_spent_ms": stats.time_spent_ms,
        "is_cpp": False,
    }
    return py_act, score, stats_dict
