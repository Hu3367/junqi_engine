"""C++ 原生核心 (junqi_core) 与 Python 双轨热拔插网桥。

设计原则：
1. 自动探测：动态检测 C++ 原生动态库 `junqi_core` 是否已编译并可用；
2. 透明降级：若 `junqi_core` 未编译，系统自动、无缝降级为纯 Python 引擎，绝不阻断任何上层业务、测试与 GUI；
3. 统一调度：提供统一的 `search_apk_fast` 与状态转换接口，自动激活 C++ 极致算力（50x~100x 加速）。
"""
from __future__ import annotations

import logging
import struct
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


# ------------------------------------------------------- 专家评估函数快路径（切片 1）

# C++ evaluate_expert 曾抛异常时永久关闭快路径，避免每次调用都付一次 try/except
# 与日志代价（热路径单局数十万次调用）。
_CPP_EVAL_FAILED = False

# 颜色 → C++ 位编码（0 = RED/"r"，1 = BLUE/"b"）
_COLOR_BITS = {"r": 0, "orange": 0, "b": 1, "purple": 1}


def _color_code(c: Optional[str]) -> int:
    return _COLOR_BITS.get(c, -1) if c is not None else -1


def encode_board(state: GameState) -> bytes:
    """棋盘 → 60 字节紧凑编码。

    每格一字节：``rank | (color << 5) | (revealed ? 0x40 : 0)``。
    rank 取值 0..22，占 5 bit；空格 rank = 0。

    注意：**只传内容，不传 dict 迭代序**。C++ 侧按 idx 升序遍历，Python 侧按
    board dict 插入序。实测两者差异 max_abs = 5.7e-14（default 标度）/
    2.3e-13（apk 标度），远低于验收阈值 1e-9 —— 依据
    ``scratch/probe_eval_order_sensitivity.py``。
    """
    buf = bytearray(60)
    for (r, c), pc in state.board.items():
        buf[r * 5 + c] = int(pc.rank) | (_COLOR_BITS.get(pc.color, 0) << 5) | \
                         (0x40 if pc.revealed else 0)
    return bytes(buf)


def encode_dead(state: GameState) -> bytes:
    """阵亡子 → 每子一字节（语法同 encode_board，无 revealed 位）。"""
    return bytes(int(pc.rank) | (_COLOR_BITS.get(pc.color, 0) << 5)
                 for pc in state.dead)


def make_cpp_weights(w) -> Any:
    """Python EvalWeights → C++ ExpertWeights。

    构建成本约数微秒，**必须由调用方缓存**（ExpertSearchEngine 在 __init__ 里
    构建一次）。C++ 侧 piece 表以 Rank 枚举值为下标。
    """
    cw = junqi_core.ExpertWeights()
    for rk, v in w.piece.items():
        cw.set_piece(int(rk), float(v))
    cw.camp_occ = float(w.camp_occ)
    cw.hq_locked = float(w.hq_locked)
    cw.flag_exposed = float(w.flag_exposed)
    cw.threat = float(w.threat)
    cw.attack = float(w.attack)
    cw.attack_camp = float(w.attack_camp)
    cw.camp_siege = float(w.camp_siege)
    cw.camp_zone = float(getattr(w, "camp_zone", 2.0))
    cw.fortress = float(w.fortress)
    cw.hidden_tempo = float(w.hidden_tempo)
    cw.echelon_si_compensation = float(getattr(w, "echelon_si_compensation", 18.0))
    cw.camp_matrix_weight = float(getattr(w, "camp_matrix_weight", 12.0))
    cw.mine_flag_guard_bonus = float(getattr(w, "mine_flag_guard_bonus", 0.0))
    cw.bomb_ratio = float(getattr(w, "bomb_ratio", 1.0 / 3.0))
    cw.use_dynamic_bomb = bool(getattr(w, "use_dynamic_bomb", False))
    return cw


def eval_expert_auto(
    state: GameState,
    seat: int,
    w=None,
    ignore_rule_draw: bool = False,
    cpp_w: Optional[Any] = None,
) -> float:
    """统一专家估值入口：优先 C++，失败/未编译则透明降级为 Python。

    等价于 ``eval_expert.evaluate_expert(state, seat, w, ignore_rule_draw)``。

    参数 ``cpp_w`` 为 ``make_cpp_weights(w)`` 的缓存结果；不传则每次现场构建
    （仅用于一次性调用/测试，搜索热路径必须传，否则每次多付数微秒）。
    """
    global _CPP_EVAL_FAILED

    if w is None:
        from .config import EvalWeights
        w = EvalWeights()

    if HAS_CPP_CORE and not _CPP_EVAL_FAILED:
        try:
            if cpp_w is None:
                cpp_w = make_cpp_weights(w)
            return junqi_core.eval_expert_cpp(
                encode_board(state),
                encode_dead(state),
                state.turn,
                state.ply,
                state.quiet,
                _color_code(state.seat_color.get(0)),
                _color_code(state.seat_color.get(1)),
                bool(state.cfg.flag_needs_mines_cleared),
                bool(state.cfg.flag_gong_only),
                bool(state.cfg.hq_locks_pieces),
                int(state.cfg.no_capture_draw_plies),
                seat,
                cpp_w,
                bool(ignore_rule_draw),
            )
        except Exception as e:  # noqa: BLE001 - 任何异常都必须退回 Python 实现
            _CPP_EVAL_FAILED = True
            logger.warning("C++ evaluate_expert failed, permanently falling back "
                           "to Python: %s", e)

    from .eval_expert import evaluate_expert
    return evaluate_expert(state, seat, w, ignore_rule_draw)


# ------------------------------------------------------- QSearch 快路径（切片 2）

# blob 头部布局，必须与 src_cpp/src/expert_qsearch.cpp::board_from_blob 严格一致：
#   "<7hi" = turn, ply, quiet, seat0, seat1, flags, no_capture_draw_plies + max_plies
_BLOB_HEADER = struct.Struct("<7hi")

_F_MINE_CLEARED = 1
_F_ALL_FLIPPED = 2
_F_GONG_ONLY = 4
_F_SUICIDE = 8
_F_HQ_LOCK = 16
_F_ENG_TURN = 32
_F_ENG_FLY = 64
_F_FIRST_FLIP = 128


def encode_state_blob(state: GameState) -> bytes:
    """整个 GameState → 单个紧凑 bytes（供 qsearch_blob 一次性跨语言传入）。

    布局：[0,60) 每格一字节；[60,60+nd) 阵亡子；尾部 18 字节头部。
    刻意用**一个** bytes 而不是多个标量参数：pybind 每个参数都有转换开销，
    18 个标量约 2 µs，打包后只剩一次对象转换。
    """
    dead = state.dead
    nd = len(dead)
    buf = bytearray(60 + nd + 18)
    for pos, pc in state.board.items():
        buf[pos[0] * 5 + pos[1]] = int(pc.rank) | (_COLOR_BITS.get(pc.color, 0) << 5) | \
                                   (0x40 if pc.revealed else 0)
    k = 60
    for pc in dead:
        buf[k] = int(pc.rank) | (_COLOR_BITS.get(pc.color, 0) << 5)
        k += 1

    cfg = state.cfg
    flags = 0
    if cfg.flag_needs_mines_cleared:
        flags |= _F_MINE_CLEARED
    if cfg.flag_needs_all_flipped:
        flags |= _F_ALL_FLIPPED
    if cfg.flag_gong_only:
        flags |= _F_GONG_ONLY
    if cfg.allow_suicide_attack:
        flags |= _F_SUICIDE
    if cfg.hq_locks_pieces:
        flags |= _F_HQ_LOCK
    if cfg.engineer_rail_turns:
        flags |= _F_ENG_TURN
    if cfg.engineer_can_fly_over_pieces:
        flags |= _F_ENG_FLY
    if state.first_flip_done:
        flags |= _F_FIRST_FLIP

    _BLOB_HEADER.pack_into(buf, k, state.turn, state.ply, state.quiet,
                           _color_code(state.seat_color.get(0)),
                           _color_code(state.seat_color.get(1)),
                           flags, int(cfg.no_capture_draw_plies), int(cfg.max_plies))
    return bytes(buf)


def make_cpp_qsearch(w, qsearch_depth: int) -> Any:
    """构造 C++ QSearch 引擎（按 ExpertSearchEngine 的权重与深度）。"""
    qs = junqi_core.ExpertQSearch()
    qs.weights = make_cpp_weights(w)
    qs.qsearch_depth = qsearch_depth
    return qs


def qsearch_auto(
    state: GameState,
    alpha: float,
    beta: float,
    depth_left: int,
    cpp_qs: Optional[Any] = None,
) -> Tuple[float, int]:
    """统一 QSearch 入口：优先 C++，返回 ``(value, qnodes)``。

    ``cpp_qs`` 为 ``make_cpp_qsearch`` 的结果（跨调用复用，保留统计）。
    C++ 不可用时返回 ``(None, 0)`` 由调用方走 Python 路径。
    """
    if HAS_CPP_CORE and cpp_qs is not None:
        try:
            before = cpp_qs.qnodes
            val = cpp_qs.qsearch_blob(encode_state_blob(state),
                                      float(alpha), float(beta), int(depth_left))
            return val, cpp_qs.qnodes - before
        except Exception as e:  # noqa: BLE001
            logger.warning("C++ qsearch failed, falling back to Python: %s", e)
    return None, 0


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
