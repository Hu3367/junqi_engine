import json
import random
import pytest

from junqi.state import GameState, Piece, Action, deal
from junqi.config import RuleConfig, EvalWeights
from junqi.rules import Rank, CAMPS, HQS, COMPOSITION
from junqi.analysis import is_dead_draw
from junqi.eval_expert import evaluate_expert
from junqi.hybrid_engine import HybridDecisionEngine


def test_game_152706_ply_152_dead_draw():
    """验证 2026-09-06 实战对局在 152 手被准确判定为必和死局，估值为 0.0。"""
    with open("games/game_20260906_152706.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = RuleConfig()
    st = deal(random.Random(data["seed"]), cfg)
    for m in data["moves"][:152]:
        frm = tuple(m["frm"])
        if m["kind"] == "flip":
            st = st.apply(Action("flip", frm))
        else:
            st = st.apply(Action("move", frm, tuple(m["to"])))

    is_draw, reason = is_dead_draw(st)
    assert is_draw is True
    assert "cannot_annihilate" in reason

    # 专家估值必须截断为 0.0
    w = EvalWeights()
    assert evaluate_expert(st, seat=0, w=w) == 0.0
    assert evaluate_expert(st, seat=1, w=w) == 0.0

    # 混合引擎 evaluate_position 也应输出 draw=1.0
    engine = HybridDecisionEngine(model_path=None)
    info = engine.evaluate_position(st)
    assert info["draw"] == 1.0
    assert info["win"] == 0.0
    assert info["loss"] == 0.0


def test_1v1_camp_and_bottom_line_dead_draw():
    """验证 1v1 单大子追单小子中，防守方在行营或底线安全走廊时被判定为必和。"""
    cfg = RuleConfig()
    # 蓝排长在底线 (0, 2)，红司令在二线 (1, 2)
    board = {
        (1, 2): Piece("r", Rank.SI, revealed=True),
        (0, 2): Piece("b", Rank.PAI, revealed=True),
    }
    dead = []
    # 双方其余 24 颗子均已阵亡
    for clr in ("r", "b"):
        for rk, count in COMPOSITION.items():
            needed = count - (1 if (clr == "r" and rk == Rank.SI) or (clr == "b" and rk == Rank.PAI) else 0)
            for _ in range(needed):
                dead.append(Piece(clr, rk, revealed=True))

    st = GameState(board=board, dead=dead, seat_color={0: "r", 1: "b"}, turn=0, cfg=cfg)
    is_draw, reason = is_dead_draw(st)
    assert is_draw is True
    assert "1v1" in reason
    assert evaluate_expert(st, seat=0) == 0.0


def test_normal_opening_not_dead_draw():
    """验证正常开局（暗子较多）绝不误判为必和。"""
    st = deal(random.Random(12345), RuleConfig())
    # 翻开一颗子定色
    st = st.apply(Action("flip", (2, 2)))
    is_draw, reason = is_dead_draw(st)
    assert is_draw is False
    assert reason == "too_many_hidden"


def test_active_midgame_with_engineers_not_dead_draw():
    """验证双方均有工兵和常规子力的正常中盘不会被误判。"""
    cfg = RuleConfig()
    board = {
        (5, 2): Piece("r", Rank.SI, revealed=True),
        (5, 0): Piece("r", Rank.GONG, revealed=True),
        (6, 2): Piece("b", Rank.JUN, revealed=True),
        (6, 4): Piece("b", Rank.GONG, revealed=True),
        (0, 1): Piece("r", Rank.QI, revealed=True),
        (11, 1): Piece("b", Rank.QI, revealed=True),
    }
    st = GameState(board=board, dead=[], seat_color={0: "r", 1: "b"}, turn=0, cfg=cfg)
    is_draw, _ = is_dead_draw(st)
    assert is_draw is False
