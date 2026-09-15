"""P0/P3：MCTS 树内重复阈值与推理热路径去冗余（审查 C6 / P2 / P3）。

  C6 `mcts.py` 树内重复判和阈值硬编码 `>= 3`，未读 `cfg.repetition_draw_count`；
     自博弈生成夹具已把判和放宽到 4（GENERATION_CFG），搜索侧却仍是 3，
     两侧口径不一致会让"搜索规避的循环"与"对局判和的循环"对不上。
  P2  `net.predict_state` 内部把 `state.legal_actions()` 算了两遍
     （一次生成掩码、一次组装 policy 字典）；MCTS 叶子评估 batch=1，
     每个叶子还要再算一次合法动作。
  P3  `hybrid_engine._price_actions` 对每个候选动作先 `state.apply(a)` 定价、
     再 `state.apply(a)` 算重复局面键，白白多构造一次完整状态。
"""
from __future__ import annotations

import unittest
from unittest import mock

import torch

from junqi.config import RuleConfig
from junqi.mcts import tree_repetition_limit
from junqi.net import JunqiNet
from junqi.state import GameState, deal


def _state(seed: int = 5) -> GameState:
    import random as _r
    return deal(_r.Random(seed), None)


class TestTreeRepetitionLimit(unittest.TestCase):
    """C6：阈值必须跟随对局规则配置。"""

    def test_default_matches_rule_config(self):
        self.assertEqual(tree_repetition_limit(_state()), 3)

    def test_follows_generation_fixture(self):
        cfg = RuleConfig(repetition_draw_count=4)
        st = _state()
        st.cfg = cfg
        self.assertEqual(tree_repetition_limit(st), 4,
                         "生成夹具把循环判和放宽到 4，搜索侧必须同步")

    def test_clamped_to_at_least_two(self):
        st = _state()
        st.cfg = RuleConfig(repetition_draw_count=1)
        self.assertGreaterEqual(tree_repetition_limit(st), 2,
                                "阈值 1 会让任何回访立刻判和，必须夹到 ≥2")

    def test_missing_cfg_falls_back_to_three(self):
        class _Bare:
            cfg = None

        self.assertEqual(tree_repetition_limit(_Bare()), 3)

    def test_mcts_actually_consults_it(self):
        """接线验证：MCTS 必须调用该 helper，而不是写死 3。"""
        import junqi.mcts as mcts_mod

        st = _state()
        net = JunqiNet(num_blocks=1, channels=8)
        calls = []
        real = mcts_mod.tree_repetition_limit

        def spy(s):
            calls.append(s)
            return real(s)

        with mock.patch.object(mcts_mod, "tree_repetition_limit", spy):
            mcts = mcts_mod.MCTS(net, simulations=2, device="cpu")
            try:
                mcts.search(st, temperature=1.0)
            except Exception:            # noqa: BLE001 - 只关心是否被调用
                pass
        self.assertTrue(calls, "MCTS 必须调用 tree_repetition_limit（不得硬编码）")


class TestPredictStateNoRedundantLegalActions(unittest.TestCase):
    """P2：单次推理只应生成一次合法动作列表。"""

    def test_legal_actions_computed_once(self):
        import junqi.state as state_mod

        st = _state(9)
        net = JunqiNet(num_blocks=1, channels=8)
        net.eval()

        real = GameState.legal_actions
        calls = []

        def counting(self):
            calls.append(1)
            return real(self)

        with mock.patch.object(GameState, "legal_actions", counting):
            net.predict_state(st, device="cpu")
        self.assertEqual(len(calls), 1,
                         f"predict_state 应只算一次 legal_actions，实际 {len(calls)} 次")

    def test_acts_can_be_supplied_by_caller(self):
        """MCTS 叶子已经算过合法动作，应可直接复用。"""
        st = _state(9)
        net = JunqiNet(num_blocks=1, channels=8)
        net.eval()
        acts = st.legal_actions()

        real = GameState.legal_actions
        calls = []

        def counting(self):
            calls.append(1)
            return real(self)

        with mock.patch.object(GameState, "legal_actions", counting):
            net.predict_state(st, device="cpu", acts=acts)
        self.assertEqual(len(calls), 0, "已提供 acts 时不得再遍历棋盘")


class TestMctsLeafReusesLegalActions(unittest.TestCase):
    """P2：MCTS 叶子评估应把已算出的合法动作传给网络。"""

    def test_predict_state_receives_acts(self):
        import junqi.mcts as mcts_mod

        st = _state(13)
        net = JunqiNet(num_blocks=1, channels=8)
        seen = []

        real_predict = JunqiNet.predict_state

        def spy(self, state, seat=None, world=None, history_counts=None,
                device="cpu", acts=None, mask=None):
            seen.append(acts)
            return real_predict(self, state, seat=seat, world=world,
                                history_counts=history_counts, device=device,
                                acts=acts, mask=mask)

        with mock.patch.object(JunqiNet, "predict_state", spy):
            mcts = mcts_mod.MCTS(net, simulations=3, device="cpu")
            mcts.search(st, temperature=1.0)

        self.assertTrue(seen, "MCTS 必须调用 predict_state")
        self.assertTrue(all(a is not None for a in seen),
                        "每个叶子都应复用已算出的合法动作列表")


class TestHybridNoDuplicateApply(unittest.TestCase):
    """P3：定价循环里每个动作只应构造一次后继状态。"""

    def test_apply_called_once_per_candidate(self):
        src = open("junqi/hybrid_engine.py", encoding="utf-8").read()
        body = src[src.index("def _price_actions"):]
        body = body[:body.index("def ", 10)] if "def " in body[10:] else body
        # 定价分支已经算出 nxt，重复键必须复用它而不是再 apply 一次
        self.assertIn("position_key(nxt)", body,
                      "重复局面判定应复用已构造的 nxt，不得再次 state.apply(a)")


if __name__ == "__main__":
    unittest.main()
