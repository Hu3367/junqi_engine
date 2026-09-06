"""验证 P1 阶段实战战术缺陷修复：
1. 空行营安全推进中继 (截图第 19 手师长走 (6,1)->(6,2)->中营)
2. 行营阻断守护与严惩盲目弃营送死 (第 231 手排长卡营堵师长严禁走 (2,3)->(2,2))
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest
from junqi.ai import ExpertAgent
from junqi.config import EvalWeights, RuleConfig, SearchConfig
from junqi.eval_expert import evaluate_expert
from junqi.rules import Rank
from junqi.search import ExpertSearchEngine
from junqi.state import Action, GameState, deal


GAME_REPLAY_PATH = Path("games/game_20260906_230423.json")


def _load_game_state_at_ply(target_ply: int) -> GameState:
    assert GAME_REPLAY_PATH.exists(), f"找不到复盘文件: {GAME_REPLAY_PATH}"
    with open(GAME_REPLAY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 该局复盘产生于 70 步无吃子限步规则下，重放切片需使用产生时的规则避免中途截断
    replay_cfg = RuleConfig(no_capture_draw_plies=70)
    st = deal(random.Random(data["seed"]), replay_cfg)
    for m in data["moves"]:
        if m["ply"] == target_ply:
            break
        if m["kind"] == "flip":
            st = st.apply(Action("flip", tuple(m["frm"])))
        else:
            st = st.apply(Action("move", tuple(m["frm"]), tuple(m["to"])))
    st.cfg = RuleConfig()
    return st


def test_ply_19_camp_staging_shi_advances():
    """测试截图第 19 手：红方师长位于 (6,1) 时，必须优先选择安全推进中继 (6,1)->(6,2)。"""
    st = _load_game_state_at_ply(19)
    assert st.turn == 1, "第 19 手应为红方 AI 执子"
    p61 = st.board.get((6, 1))
    assert p61 is not None and p61.rank == Rank.SHI and p61.color == "r"

    agent = ExpertAgent(search=SearchConfig(depth=3), seed=477429043)
    scored = agent.choose_actions(st, topn=3)
    assert len(scored) > 0

    best_action = scored[0][0]
    assert best_action.kind == "move", f"第 19 手最优走法应为移动而非盲目翻棋，实际为: {best_action}"
    assert best_action.frm == (6, 1) and best_action.to == (6, 2), (
        f"第 19 手师长应走向安全中继站 (6,2)，实际走法为: {best_action}"
    )


def test_ply_231_anti_camp_abandonment():
    """测试第 231 手：红方排长在 (2,3) 行营内卡守蓝方师长时，严禁自杀弃营走 (2,3)->(2,2)。"""
    st = _load_game_state_at_ply(231)
    assert st.turn == 1, "第 231 手应为红方 AI 执子"
    p23 = st.board.get((2, 3))
    assert p23 is not None and p23.rank == Rank.PAI and p23.color == "r"
    p34 = st.board.get((3, 4))
    assert p34 is not None and p34.rank == Rank.SHI and p34.color == "b"

    agent = ExpertAgent(search=SearchConfig(depth=3), seed=477429043)
    best_action = agent.select_action(st)

    suicide_abandon = Action("move", (2, 3), (2, 2))
    assert best_action != suicide_abandon, (
        f"第 231 手严禁选择将守营排长移出至 (2,2) 拱手让营并被师长扑杀！实际选择为: {best_action}"
    )
    # 验证最优走法是安全进驻空营 (7,2)->(7,3)
    assert best_action == Action("move", (7, 2), (7, 3)), (
        f"第 231 手最优决策应为挺进空营 (7,2)->(7,3)，实际决策为: {best_action}"
    )

    scored = agent.choose_actions(st, topn=4)
    top4_actions = [a for a, _ in scored]
    assert suicide_abandon not in top4_actions, (
        f"第 231 手自杀弃营走法绝不应出现在前 4 候选走法中！前 4 走法为: {top4_actions}"
    )


def test_camp_staging_heuristic_scoring():
    """测试大子向空营推进的启发评分显著高于普通静步与翻棋。"""
    st = _load_game_state_at_ply(19)
    eng = ExpertSearchEngine()

    staging_move = Action("move", (6, 1), (6, 2))
    flip_move = Action("flip", (1, 3))
    quiet_move = Action("move", (8, 3), (8, 4)) if (8, 4) not in st.board else None

    s_staging = eng._score_action(staging_move, st, 0, None)
    s_flip = eng._score_action(flip_move, st, 0, None)

    assert s_staging > s_flip, f"大子安全挺进空营启发分 ({s_staging}) 必须高于翻棋分 ({s_flip})"
    assert s_staging >= 200_000.0, f"大子中继推进优先级应达到 200,000+，实际为: {s_staging}"


def test_anti_camp_abandonment_penalty():
    """测试营内弱子在敌大子近身窥视时出营被判定为严厉负分。"""
    st = _load_game_state_at_ply(231)
    eng = ExpertSearchEngine()

    suicide_abandon = Action("move", (2, 3), (2, 2))
    s_abandon = eng._score_action(suicide_abandon, st, 0, None)
    assert s_abandon <= -300_000.0, f"在敌师长贴营窥视下出营送死必须获得严厉惩罚分，实际为: {s_abandon}"
