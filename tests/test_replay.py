"""复盘 .sav 解码器测试。真实样本依赖 ../军旗复盘/，不存在则跳过。"""
from __future__ import annotations

import os
import unittest

from junqi.config import RuleConfig
from junqi.replay import (SPECIAL_EVENT, board_from_table, cell_rc, parse_sav,
                          replay_sav)
from junqi.rules import CAMPS, Rank
from junqi.state import Piece

REPLAY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "军旗复盘")

SAMPLE = "251124202204 棋手 62240-棋手 38285.sav"   # 曾暴露工兵飞行差异的一局


@unittest.skipUnless(os.path.isdir(REPLAY_DIR), "无复盘数据目录，跳过")
class TestSavDecode(unittest.TestCase):
    def test_parse_header_and_table(self):
        g = parse_sav(os.path.join(REPLAY_DIR, SAMPLE))
        self.assertEqual(g.version, 1)
        self.assertEqual(len(g.table), 60)
        # 身份表：0 恰为 10 个行营；每色 25 子构成标准配伍
        cells0 = [i for i, v in enumerate(g.table) if v == 0]
        self.assertEqual(len(cells0), 10)
        self.assertEqual({cell_rc(i) for i in cells0},
                         {p for p in CAMPS})
        for lo, hi in ((1, 12), (13, 24)):
            from collections import Counter
            cnt = Counter(v - lo for v in g.table if lo <= v <= hi)
            self.assertEqual(sum(cnt.values()), 25)
            self.assertEqual(cnt[0], 1)             # 每色 1 面司令
            self.assertEqual(cnt[11], 1)            # 每色 1 面军旗
        # 每色 1 面军旗；翻棋模式旗随机放置（数据证实不要求在大本营，
        # "军旗必须放大本营"仅是布局模式规则）
        flags = [v for v in g.table if v in (12, 24)]
        self.assertEqual(len(flags), 2)

    def test_replay_full_game(self):
        cfg = RuleConfig(engineer_can_fly_over_pieces=True)
        g = replay_sav(parse_sav(os.path.join(REPLAY_DIR, SAMPLE)), cfg=cfg)
        self.assertTrue(g.replay_ok, g.error)
        self.assertGreater(g.n_plies, 0)

    def test_special_event_truncates(self):
        # 含 (255,255,1) 的局应截断且不报错
        found = False
        for name in sorted(os.listdir(REPLAY_DIR))[:200]:
            if not name.endswith(".sav"):
                continue
            g = parse_sav(os.path.join(REPLAY_DIR, name))
            if SPECIAL_EVENT in g.moves:
                g2 = replay_sav(g)
                self.assertTrue(g2.replay_ok)
                self.assertTrue(g2.stopped_on_event)
                self.assertIsNone(g2.error)
                found = True
                break
        self.assertTrue(found, "前 200 局中未找到特殊事件样本")


class TestBoardFromTable(unittest.TestCase):
    def test_piece_mapping(self):
        table = [0] * 60
        table[0] = 1        # 色A 司令
        table[2] = 12       # 色A 军旗
        table[59] = 13      # 色B 司令
        board = board_from_table(table)
        self.assertEqual(board[(0, 0)], Piece("r", Rank.SI))
        self.assertEqual(board[(0, 2)], Piece("r", Rank.QI))
        self.assertEqual(board[(11, 4)], Piece("b", Rank.SI))


if __name__ == "__main__":
    unittest.main()
