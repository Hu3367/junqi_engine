"""P1 阶段行营往复互窜根治与据点辐射拓荒专项回归测试。

针对 games/game_20260907_211452.json 与 games/game_20260907_211613.json 暴露的
营间无休止互窜（如 (3,2)<->(2,3) 往复）与无故出营闲走缺陷进行定向回归验证。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from junqi.ai import ApkNativeAgent, ExpertAgent
from junqi.config import EvalWeights, RuleConfig, SearchConfig
from junqi.eval_expert import evaluate_expert
from junqi.rules import Rank, is_camp
from junqi.search import ExpertSearchEngine
from junqi.state import Action, GameState, Piece, deal


def test_camp_shuttle_negative_ordering():
    """测试非吃子跨营移动 (camp-to-camp shuttle) 在走法排序中被严厉压制为负优先级。"""
    st = GameState(
        board={
            (3, 2): Piece("r", Rank.TUAN, revealed=True),
            (3, 1): Piece("r", Rank.LIAN, revealed=False),
        },
        seat_color={0: "b", 1: "r"},
        turn=1,
        first_flip_done=True,
    )
    eng = ExpertSearchEngine()

    # (3,2) 为中营，(2,3) 为角营，两者皆为行营
    shuttle_act = Action(kind="move", frm=(3, 2), to=(2, 3))
    flip_act = Action(kind="flip", frm=(3, 1))

    score_shuttle = eng._score_action(shuttle_act, st, 0, None)
    score_flip = eng._score_action(flip_act, st, 0, None)

    assert score_shuttle <= -150_000.0, f"营间闲走必须被赋予严重负优先级，实际为: {score_shuttle}"
    assert score_flip > 100_000.0, f"驻营据点邻接翻棋应享有辐射拓荒高优先级，实际为: {score_flip}"
    assert score_flip > score_shuttle, "邻营拓荒翻棋优先级必须远高于营间互窜闲走"


def test_unique_empty_camp_attribution():
    """测试单一棋子不得同时虚报多个空营的控制权（杜绝单子出营刷分虚胀）。"""
    # 红团长在十字路口 (2,2)，周围接 (2,1), (2,3), (3,2) 三个空营
    st_cross = GameState(
        board={(2, 2): Piece("r", Rank.TUAN, revealed=True)},
        seat_color={0: "b", 1: "r"},
        turn=1,
        first_flip_done=True,
    )
    # 红团长稳坐中营 (3,2)
    st_camp = GameState(
        board={(3, 2): Piece("r", Rank.TUAN, revealed=True)},
        seat_color={0: "b", 1: "r"},
        turn=1,
        first_flip_done=True,
    )

    w = EvalWeights.apk_weights()
    eval_cross = evaluate_expert(st_cross, 1, w)
    eval_camp = evaluate_expert(st_camp, 1, w)

    # 稳坐中营的据点价值必须显著高于脱营暴露在十字路口
    assert eval_camp > eval_cross + 50.0, (
        f"驻守核心中营估值 ({eval_camp}) 必须大幅高于脱营闲走 ({eval_cross})"
    )


def test_game_211613_ply5_flip_preference():
    """回归验证 games/game_20260907_211613.json 第 5 手：

    红方占领中营 (3,2) 后，严禁向空角营 (2,3) 或 (4,1) 往复串门，
    必须坚守中营据点并优先翻开周围暗子 (3,1), (3,3), (4,2)。
    """
    game_file = Path("games/game_20260907_211613.json")
    assert game_file.exists()
    with open(game_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    st = deal(random.Random(data["seed"]))
    for m in data["moves"]:
        if m["ply"] == 5:
            break
        act = Action(frm=tuple(m["frm"]), to=tuple(m["to"]) if m["to"] else None, kind=m["kind"])
        st = st.apply(act)

    assert st.turn == 1, "第 5 手应为席位 1 行动"
    p32 = st.board.get((3, 2))
    assert p32 is not None and is_camp((3, 2))

    # 1. 验证 ExpertAgent
    expert = ExpertAgent(search=SearchConfig(depth=3), seed=42)
    best_act_exp = expert.select_action(st)
    assert best_act_exp.kind == "flip", (
        f"ExpertAgent 在第 5 手应优先翻开据点周围暗子辐射拓荒，实际为: {best_act_exp}"
    )
    assert best_act_exp.frm in ((3, 1), (3, 3), (4, 2)), (
        f"ExpertAgent 翻棋目标应在中营辐射邻域，实际为: {best_act_exp.frm}"
    )

    # 2. 验证 ApkNativeAgent
    apk = ApkNativeAgent(level="advanced", seed=42)
    best_act_apk = apk.select_action(st)
    assert best_act_apk.kind == "flip", (
        f"ApkNativeAgent 在第 5 手应优先翻开据点周围暗子辐射拓荒，实际为: {best_act_apk}"
    )
    assert best_act_apk.frm in ((3, 1), (3, 3), (4, 2)), (
        f"ApkNativeAgent 翻棋目标应在中营辐射邻域，实际为: {best_act_apk.frm}"
    )


def test_game_211452_ply7_stay_and_flip():
    """回归验证 games/game_20260907_211452.json 第 7 手：

    蓝方占领下中营 (8,2) 后，严禁无故出营走 (8,2)->(9,1)，
    必须坚守据点翻开邻营暗子 (8,1) 实施辐射拓荒。
    """
    game_file = Path("games/game_20260907_211452.json")
    assert game_file.exists()
    with open(game_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    st = deal(random.Random(data["seed"]))
    for m in data["moves"]:
        if m["ply"] == 7:
            break
        act = Action(frm=tuple(m["frm"]), to=tuple(m["to"]) if m["to"] else None, kind=m["kind"])
        st = st.apply(act)

    assert st.turn == 1, "第 7 手应为席位 1 (蓝方) 行动"
    p82 = st.board.get((8, 2))
    assert p82 is not None and is_camp((8, 2))

    # 1. 验证 ExpertAgent
    expert = ExpertAgent(search=SearchConfig(depth=3), seed=42)
    best_act_exp = expert.select_action(st)
    assert best_act_exp.kind == "flip", (
        f"ExpertAgent 第 7 手严禁弃营闲走，应优先翻棋开拓，实际为: {best_act_exp}"
    )
    assert best_act_exp.frm == (8, 1), f"ExpertAgent 应首选翻开中营邻近暗子 (8,1)，实际为: {best_act_exp}"

    # 2. 验证 ApkNativeAgent
    apk = ApkNativeAgent(level="advanced", seed=42)
    best_act_apk = apk.select_action(st)
    assert best_act_apk.kind == "flip", (
        f"ApkNativeAgent 第 7 手严禁弃营闲走，应优先翻棋开拓，实际为: {best_act_apk}"
    )
    assert best_act_apk.frm in ((8, 1), (8, 3)), (
        f"ApkNativeAgent 应首选翻开中营邻近暗子，实际为: {best_act_apk}"
    )
