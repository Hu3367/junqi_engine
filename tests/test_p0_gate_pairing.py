"""P0：门控配对同牌 / 设备 / 单一实现（审查 R3）。

问题：训练主循环的 `train_rl.evaluate_gate` 是一套与 `eval_gate.run_gate` 平行
的重复实现，且存在两处硬伤：
  1. 两个方向用 `seed+i` 与 `seed+100_000+i` → **非配对同牌**，先后手差异与
     发牌运气没有被消去，和棋密集规则下几乎无分辨力；
  2. `device` 硬编码 "cpu"（即便有 GPU）。
而 `eval_gate.run_gate` 本身是合格的（配对同牌 + Wilson + 三元 SPRT）。

本文件固化：训练环路必须走 eval_gate.run_gate，且配对语义正确。
"""
from __future__ import annotations

import unittest
from unittest import mock

from junqi.eval_gate import run_gate
from junqi.state import GameState, deal


def _init_json(seed: int) -> str:
    """一副可作为初始局面的局面序列化串。"""
    import random as _r
    return deal(_r.Random(seed), None).to_json()


def _fake_play_game(spec0, spec1, seed, cfg=None, model_path0=None,
                    model_path1=None, init_state=None, device=None, **kw):
    """替代 selfplay.play_game，只记录调用参数并返回一条可统计的记录。"""
    _fake_play_game.calls.append({
        "spec0": spec0, "spec1": spec1, "seed": seed,
        "model0": model_path0, "model1": model_path1,
        "init": (init_state.to_json() if init_state is not None else None), "device": device,
    })
    return {"seed": seed, "a": spec0, "b": spec1, "winner": 0,
            "reason": "flag", "final_eval0": 0.0, "final_eval1": 0.0}


class _FakeGateTest(unittest.TestCase):
    def setUp(self):
        _fake_play_game.calls = []
        self.patcher = mock.patch("junqi.selfplay.play_game", _fake_play_game)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()


class TestRunGatePairing(_FakeGateTest):

    def test_both_directions_share_same_seed_and_init_state(self):
        """配对语义：同一 seed 的先后手两局必须同种子、同初始局面，只换模型。"""
        deals = [_init_json(101), _init_json(202)]
        run_gate("nn_mcts_8", "nn_mcts_8", seeds=[5, 6], workers=1,
                 model_a="cand.pt", model_b="best.pt", init_states=deals)
        calls = _fake_play_game.calls
        self.assertEqual(len(calls), 4, "每个 seed 必须跑先后手两局")
        by_seed = {}
        for c in calls:
            by_seed.setdefault(c["seed"], []).append(c)
        self.assertEqual(sorted(by_seed), [5, 6])
        for seed, pair in by_seed.items():
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]["init"], pair[1]["init"],
                             f"seed={seed} 的先后手两局必须是同一副牌")
            self.assertIsNotNone(pair[0]["init"])
            first, second = pair
            self.assertEqual((first["model0"], first["model1"]),
                             ("cand.pt", "best.pt"))
            self.assertEqual((second["model0"], second["model1"]),
                             ("best.pt", "cand.pt"), "第二局必须交换座位")

    def test_init_states_length_mismatch_is_loud(self):
        with self.assertRaises(ValueError):
            run_gate("random", "random", seeds=[1, 2, 3], workers=1,
                     init_states=["only-one"])

    def test_device_propagates_to_every_game(self):
        run_gate("random", "random", seeds=[1], workers=1, device="cuda")
        self.assertTrue(_fake_play_game.calls)
        for c in _fake_play_game.calls:
            self.assertEqual(c["device"], "cuda", "device 必须透传，不得硬编码 cpu")

    def test_default_device_is_cpu(self):
        run_gate("random", "random", seeds=[1], workers=1)
        self.assertEqual(_fake_play_game.calls[0]["device"], "cpu")


class TestTrainLoopUsesUnifiedGate(unittest.TestCase):
    """训练主循环必须委托给 eval_gate.run_gate（不得再自建平行实现）。"""

    def _make_report(self, wins, draws, losses, reason="flag"):
        return {
            "totals": {"wins": wins, "draws": draws, "losses": losses},
            "reason_breakdown": {reason: {"wins": wins, "draws": draws,
                                          "losses": losses}},
            "n_games": wins + draws + losses,
        }

    def test_evaluate_gate_delegates_per_scene(self):
        from junqi import train_rl

        calls = []

        def fake_run_gate(spec_a, spec_b, seeds=None, workers=None,
                          model_a=None, model_b=None, init_states=None,
                          device=None, **kw):
            calls.append({"spec_a": spec_a, "spec_b": spec_b, "seeds": seeds,
                          "init_states": init_states, "model_a": model_a,
                          "model_b": model_b, "device": device,
                          "workers": workers})
            return self._make_report(1, 1, 0)

        with mock.patch.object(train_rl, "run_gate", fake_run_gate), \
             mock.patch.object(train_rl, "_load_eval_jsons",
                               lambda stage, limit: ["j0", "j1"][:limit]), \
             mock.patch.object(train_rl, "GATE_STAGES", ("opening", "midgame")):
            stage_stats, ref_score = train_rl.evaluate_gate(
                "cand.pt", "best.pt", sims=8, games_per_side=2, seed=100,
                workers=3, ref_games=0, device="cuda")

        self.assertEqual(len(calls), 3, "opening / midgame / random 各一次")
        for c in calls:
            self.assertEqual(c["model_a"], "cand.pt")
            self.assertEqual(c["model_b"], "best.pt")
            self.assertEqual(c["device"], "cuda", "设备必须透传，不得写死 cpu")
            self.assertEqual(c["workers"], 3)
            # 每个 seed 对应一副初始局面：seeds 数量 == init_states 数量
            self.assertEqual(len(c["seeds"]), len(c["init_states"]))
            # 配对语义：每个种子只发射一次，先后手由 run_gate 内部互换座位实现
            self.assertEqual(max(c["seeds"]) - min(c["seeds"]),
                             len(c["seeds"]) - 1)

        self.assertIn("overall", stage_stats)
        self.assertEqual(stage_stats["overall"]["wins"], 3)
        self.assertEqual(stage_stats["overall"]["games"], 6)
        self.assertIsNone(ref_score)

    def test_reference_run_also_paired(self):
        from junqi import train_rl

        calls = []

        def fake_run_gate(spec_a, spec_b, seeds=None, **kw):
            calls.append({"spec_a": spec_a, "spec_b": spec_b,
                          "seeds": list(seeds), "kw": kw})
            return self._make_report(1, 0, 1)

        with mock.patch.object(train_rl, "run_gate", fake_run_gate), \
             mock.patch.object(train_rl, "_load_eval_jsons",
                               lambda stage, limit: ["j0", "j1", "j2"][:limit]), \
             mock.patch.object(train_rl, "GATE_STAGES", ("opening",)):
            _, ref_score = train_rl.evaluate_gate(
                "cand.pt", "best.pt", sims=8, games_per_side=0, seed=9,
                workers=1, ref_games=2)

        ref_calls = [c for c in calls if c["spec_b"] == "search2"]
        self.assertTrue(ref_calls, "参考对抗也必须走统一门控")
        self.assertIsNotNone(ref_score)


if __name__ == "__main__":
    unittest.main()
