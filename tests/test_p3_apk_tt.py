"""P3：apk_engine 置换表正确性与内存上界（审查 P6）。

问题（`junqi/apk_engine.py`）：
  1. `_get_hash()` 只返回 `compute_zobrist(state) & tt_mask`（默认 18 位），
     写入与读取都用这个**截断值**当键，从不校验完整 Zobrist；
     跨局面哈希碰撞会直接返回**错误分数与错误 PV**（棋力随机抖动，且不报错）。
     对比 `junqi/tt.py` 有 `entry.key != key` 校验，apk 版缺失。
  2. `self.tt` 是无界 `dict`，`tt_size_power` 只做掩码不做容量限制，
     全库无 `clear()`——长程自对弈下随局面数线性膨胀（内存泄漏）。
"""
from __future__ import annotations

import unittest

from junqi.apk_engine import ApkSearchEngine
from junqi.rules import ROWS, COLS


class TestTtKeyValidation(unittest.TestCase):
    """P6-1：命中必须校验完整 Zobrist，不得只凭截断哈希。"""

    def _engine(self, power: int = 4, max_entries: int = 64) -> ApkSearchEngine:
        return ApkSearchEngine(tt_size_power=power, tt_max_entries=max_entries,
                               seed=1)

    def test_colliding_entry_is_rejected(self):
        eng = self._engine()
        key_a = 0b1010
        key_b = key_a + (1 << 20)          # 与 key_a 同低位（掩码后同槽），但完整键不同
        self.assertEqual(key_a & eng.tt_mask, key_b & eng.tt_mask,
                         "构造前提：两键必须落在同一槽位")
        eng._tt_store(key_a, depth=3, flag=1, score=123.0, move=None)
        self.assertIsNotNone(eng._tt_lookup(key_a))
        self.assertIsNone(eng._tt_lookup(key_b),
                          "不同完整键即使同槽也必须视为未命中（否则返回错误分数）")

    def test_exact_key_returns_entry(self):
        eng = self._engine()
        eng._tt_store(0xABCDE, depth=2, flag=1, score=-7.5, move=None)
        ent = eng._tt_lookup(0xABCDE)
        self.assertIsNotNone(ent)
        self.assertEqual(ent[0], 2)

    def test_deeper_entry_survives_shallower_overwrite(self):
        """经典 TT 行为：同槽同键写入不得被更浅的条目覆盖掉更深的结果。"""
        eng = self._engine()
        eng._tt_store(0x1234, depth=5, flag=1, score=10.0, move=None)
        eng._tt_store(0x1234, depth=1, flag=1, score=-99.0, move=None)
        ent = eng._tt_lookup(0x1234)
        self.assertEqual(ent[0], 5, "浅条目不得覆盖深条目")
        self.assertAlmostEqual(ent[2], 10.0)


class TestTtBounded(unittest.TestCase):
    """P6-2：置换表必须有容量上界，不能无限膨胀。"""

    def test_table_never_exceeds_capacity(self):
        eng = ApkSearchEngine(tt_size_power=4, tt_max_entries=32, seed=1)
        for i in range(500):
            eng._tt_store(i * 1_000_003, depth=2, flag=1, score=float(i), move=None)
        self.assertLessEqual(len(eng.tt), 32,
                             "置换表条目数超过上界（内存无界增长）")

    def test_default_capacity_matches_mask_bits(self):
        eng = ApkSearchEngine(tt_size_power=6, seed=1)
        self.assertEqual(eng.tt_max_entries, 1 << 6)

    def test_eviction_keeps_table_usable(self):
        eng = ApkSearchEngine(tt_size_power=4, tt_max_entries=8, seed=1)
        for i in range(40):
            eng._tt_store(i * 7919 + 13, depth=2, flag=1, score=1.0, move=None)
        eng._tt_store(999_999, depth=4, flag=1, score=42.0, move=None)
        self.assertIsNotNone(eng._tt_lookup(999_999),
                             "淘汰后必须仍能正常读写")


class TestZobristStillUsed(unittest.TestCase):
    """保证改动没有把搜索主路径的键来源换掉。"""

    def test_get_hash_is_full_deterministic_zobrist(self):
        from junqi.state import deal
        import random as _r

        eng = ApkSearchEngine(tt_size_power=8, seed=1)
        game = deal(_r.Random(3), None)
        h = eng._get_hash(game)
        self.assertEqual(h, eng._get_hash(game), "_get_hash 必须确定性")
        self.assertGreaterEqual(h, 1 << 8, "_get_hash 现在返回完整键（未截断）")

    def test_slot_is_masked(self):
        from junqi.state import deal
        import random as _r

        eng = ApkSearchEngine(tt_size_power=8, seed=1)
        game = deal(_r.Random(3), None)
        slot = eng._tt_slot(eng._get_hash(game))
        self.assertLess(slot, 1 << 8, "槽位索引必须落在掩码范围内")


if __name__ == "__main__":
    unittest.main()
