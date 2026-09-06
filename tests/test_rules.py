"""M1 规则引擎单元测试。运行：python -m unittest discover -s tests -v"""
from __future__ import annotations

import random
import unittest

from junqi.config import RuleConfig
from junqi.rules import (BOTH_DIE, ATTACKER_WINS, DEFENDER_WINS, CAMPS, HQS,
                         COMPOSITION, NEIGHBORS, PLAY_POSITIONS, RANK_CN, Rank,
                         battle, in_board, is_rail)
from junqi.state import Action, GameState, Piece, deal, position_key


def mk(board_spec, turn=0, seat_color=None, first_flip_done=True, dead=(),
       cfg=None, ply=0):
    """按 {(r,c): ('r','SI',revealed)} 快速构造状态。"""
    board = {}
    for pos, (color, rank_name, revealed) in board_spec.items():
        board[pos] = Piece(color, Rank[rank_name], revealed)
    sc = seat_color or {0: "r", 1: "b"}
    return GameState(board=board, dead=dead, seat_color=sc, turn=turn,
                     first_flip_done=first_flip_done, ply=ply,
                     cfg=cfg or RuleConfig())


def moves_of(state, frm=None):
    acts = state.legal_actions()
    return [a for a in acts if a.kind == "move" and (frm is None or a.frm == frm)]


def dests(state, frm):
    return {a.to for a in moves_of(state, frm)}


class TestGeometry(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(PLAY_POSITIONS), 50)
        self.assertEqual(len(CAMPS), 10)
        self.assertEqual(len(HQS), 4)
        self.assertFalse(CAMPS & HQS)

    def test_camps_not_on_rail(self):
        for p in CAMPS:
            self.assertFalse(is_rail(p), f"行营 {p} 不应在铁路上")

    def test_camp_has_8_neighbors(self):
        for p in CAMPS:
            self.assertEqual(len(NEIGHBORS[p]), 8, f"行营 {p} 应有 8 邻")

    def test_cross_center_connections(self):
        # 前线只有 col 0/2/4 连通
        self.assertIn((6, 0), NEIGHBORS[(5, 0)])
        self.assertIn((6, 2), NEIGHBORS[(5, 2)])
        self.assertIn((6, 4), NEIGHBORS[(5, 4)])
        self.assertNotIn((6, 1), NEIGHBORS[(5, 1)])
        self.assertNotIn((6, 3), NEIGHBORS[(5, 3)])

    def test_railway_boundaries(self):
        # 两侧纵向铁路只延伸到倒数第二排 (row 1 和 row 10)，不到达大本营底线 (row 0 和 row 11)
        self.assertTrue(is_rail((1, 0)))
        self.assertTrue(is_rail((1, 4)))
        self.assertTrue(is_rail((10, 0)))
        self.assertTrue(is_rail((10, 4)))

        self.assertFalse(is_rail((0, 0)))
        self.assertFalse(is_rail((0, 4)))
        self.assertFalse(is_rail((11, 0)))
        self.assertFalse(is_rail((11, 4)))


class TestBattle(unittest.TestCase):
    def test_regular(self):
        self.assertEqual(battle(Rank.SI, Rank.PAI), ATTACKER_WINS)
        self.assertEqual(battle(Rank.PAI, Rank.SI), DEFENDER_WINS)
        self.assertEqual(battle(Rank.TUAN, Rank.TUAN), BOTH_DIE)

    def test_bomb(self):
        self.assertEqual(battle(Rank.ZHA, Rank.SI), BOTH_DIE)
        self.assertEqual(battle(Rank.SI, Rank.ZHA), BOTH_DIE)
        self.assertEqual(battle(Rank.ZHA, Rank.LEI), BOTH_DIE)

    def test_mine(self):
        self.assertEqual(battle(Rank.GONG, Rank.LEI), ATTACKER_WINS)
        self.assertEqual(battle(Rank.PAI, Rank.LEI), DEFENDER_WINS)
        self.assertEqual(battle(Rank.SI, Rank.LEI), DEFENDER_WINS)

    def test_flag(self):
        self.assertEqual(battle(Rank.PAI, Rank.QI), ATTACKER_WINS)


class TestDeal(unittest.TestCase):
    def test_deal_composition(self):
        st = deal(random.Random(42))
        self.assertEqual(len(st.board), 50)
        for p in CAMPS:
            self.assertNotIn(p, st.board)
        from collections import Counter
        cnt = Counter((pc.color, pc.rank) for pc in st.board.values())
        for color in ("r", "b"):
            for rank, n in COMPOSITION.items():
                self.assertEqual(cnt[(color, rank)], n, f"{color}{rank}")
        self.assertIsNone(st.seat_color[0])
        self.assertEqual(st.turn, 0)
        # 首翻前只能翻子
        self.assertTrue(all(a.kind == "flip" for a in st.legal_actions()))

    def test_first_flip_assigns_color(self):
        st = deal(random.Random(7))
        act = st.legal_actions()[0]
        pc = st.board[act.frm]
        nxt = st.apply(act)
        self.assertTrue(pc.revealed is False)
        self.assertEqual(nxt.seat_color[0], pc.color)
        self.assertEqual(nxt.seat_color[1], "b" if pc.color == "r" else "r")
        self.assertTrue(nxt.first_flip_done)
        self.assertEqual(nxt.turn, 1)
        self.assertTrue(nxt.board[act.frm].revealed)

    def test_json_roundtrip(self):
        st = deal(random.Random(3))
        # 随便走几手
        rng = random.Random(1)
        cur = st
        for _ in range(12):
            acts = cur.legal_actions()
            cur = cur.apply(rng.choice(acts))
        text = cur.to_json()
        back = GameState.from_json(text)
        self.assertEqual(back.board, cur.board)
        self.assertEqual(back.seat_color, cur.seat_color)
        self.assertEqual((back.turn, back.ply, back.first_flip_done),
                         (cur.turn, cur.ply, cur.first_flip_done))
        self.assertEqual(back.dead, cur.dead)


class TestMovegen(unittest.TestCase):
    def setUp(self):
        self.cfg = RuleConfig()

    def test_rail_straight_only_no_corner_turns(self):
        # App 实测：铁路一律直线，四角弧形弯也不许拐
        st = mk({(1, 1): ("r", "SHI", True)})
        d = dests(st, (1, 1))
        expect = {(0, 1), (2, 1),          # 公路一步
                  (1, 0), (1, 2), (1, 3), (1, 4)}   # row1 铁路直线
        self.assertEqual(d, expect)        # 不再经弧线进入两侧纵线
        # 从纵线 (2,0) 北滑可直达倒数第二排角点 (1,0)，但底线 (0,0) 非铁路不可直达
        st2 = mk({(2, 0): ("r", "SHI", True)})
        d2 = dests(st2, (2, 0))
        self.assertIn((1, 0), d2)
        self.assertNotIn((0, 0), d2)       # row 0 非铁路
        self.assertNotIn((1, 1), d2)       # 不能在角点拐进 row1

    def test_cross_center_rail_blocked_at_col1_col3(self):
        # App 实测：(5,1)-(6,1)、(5,3)-(6,3) 完全不通（山界河流隔断）
        st = mk({(5, 1): ("r", "SHI", True)})
        d = dests(st, (5, 1))
        self.assertNotIn((6, 1), d)        # 阻断隔断
        self.assertIn((5, 2), d)           # row5 铁路直线正常
        # col2 中桥轨道贯通（公路与铁路均相通）
        st2 = mk({(5, 2): ("r", "SHI", True)})
        d2 = dests(st2, (5, 2))
        self.assertIn((6, 2), d2)          # 中桥直线贯通
        # 工兵可通过中桥铁路线转弯过河
        st_gong = mk({(5, 1): ("r", "GONG", True)})
        d_gong = dests(st_gong, (5, 1))
        self.assertIn((6, 2), d_gong)      # 工兵经由中桥到达 row6
        self.assertIn((6, 0), d_gong)      # 工兵经中桥可延伸至整个前线铁路
        # col0/col4 左右铁桥直通保持
        st3 = mk({(5, 0): ("r", "SHI", True)})
        self.assertIn((6, 0), dests(st3, (5, 0)))

    def test_t_junction_no_turn(self):
        # 位于前线 T 交点 (5,4)，不能通过铁路转弯离开直线方向
        st = mk({(5, 4): ("r", "SHI", True)})
        d = dests(st, (5, 4))
        self.assertIn((5, 3), d)      # 沿 row5 直行
        self.assertIn((6, 4), d)      # col4 直行（含跨前线）
        self.assertIn((4, 4), d)
        # 直行穿过 T 交点不受限：从 (3,0) 南下滑过 (5,0)/(6,0) 到达 row 4..10（row 11 非铁路）
        st2 = mk({(3, 0): ("r", "SHI", True)})
        d2 = dests(st2, (3, 0))
        for r in range(4, 11):
            self.assertIn((r, 0), d2, (r, 0))
        self.assertNotIn((11, 0), d2)

    def test_slide_blocked_by_own_and_enemy(self):
        st = mk({(1, 1): ("r", "SHI", True),
                 (1, 3): ("r", "LV", True)})
        d = dests(st, (1, 1))
        self.assertIn((1, 2), d)
        self.assertNotIn((1, 3), d)   # 己方阻挡且不可吃
        self.assertNotIn((1, 4), d)
        # 敌子可吃但不可穿越
        st2 = mk({(1, 1): ("r", "SHI", True),
                  (1, 3): ("b", "LV", True)})
        d2 = dests(st2, (1, 1))
        self.assertIn((1, 3), d2)
        self.assertNotIn((1, 4), d2)

    def test_road_one_step_and_camp_diagonal(self):
        st = mk({(1, 1): ("r", "SHI", True)})
        d = dests(st, (1, 1))
        for p in [(0, 1), (2, 1), (1, 0), (1, 2)]:
            self.assertIn(p, d, p)
        # 行营斜道：从 (1,0) 可以一步斜进 (2,1) 行营
        st2 = mk({(1, 0): ("r", "SHI", True)})
        d2 = dests(st2, (1, 0))
        self.assertIn((2, 1), d2)     # 进营合法（行营只是不可被攻击）

    def test_cannot_attack_into_camp(self):
        st = mk({(1, 0): ("r", "SI", True), (2, 1): ("b", "PAI", True)})
        d = dests(st, (1, 0))
        self.assertNotIn((2, 1), d)   # 行营内的敌子不可攻击

    def test_cannot_attack_hidden(self):
        st = mk({(1, 0): ("r", "SI", True), (1, 1): ("b", "PAI", False)})
        d = dests(st, (1, 0))
        self.assertNotIn((1, 1), d)

    def test_immovable_pieces(self):
        # 大本营默认不锁（App 实测）；开启 hq_locks_pieces 才锁
        spec = {(5, 0): ("r", "LEI", True), (5, 2): ("r", "QI", True)}
        st = mk(spec)
        self.assertEqual(dests(st, (5, 0)), set())   # 地雷不能动
        self.assertEqual(dests(st, (5, 2)), set())   # 军旗不能动
        st_hq = mk({(0, 1): ("r", "SI", True)},
                   cfg=RuleConfig(hq_locks_pieces=True))
        self.assertEqual(dests(st_hq, (0, 1)), set())   # 旧规则：锁死
        st_free = mk({(0, 1): ("r", "SI", True)})
        self.assertTrue(dests(st_free, (0, 1)))         # 翻棋实测：可自由走出大本营

    def test_no_suicide_attack(self):
        # App 实测：不允许小子撞大子
        st = mk({(0, 0): ("r", "PAI", True), (0, 1): ("b", "SI", True)})
        d = dests(st, (0, 0))
        self.assertNotIn((0, 1), d)      # 排长不可撞司令
        # 同归于尽的交换仍合法（炸弹/同级）
        st2 = mk({(0, 0): ("r", "ZHA", True), (0, 1): ("b", "SI", True)})
        self.assertIn((0, 1), dests(st2, (0, 0)))
        st3 = mk({(0, 0): ("r", "LV", True), (0, 1): ("b", "LV", True)})
        self.assertIn((0, 1), dests(st3, (0, 0)))
        # 碰雷也是自杀：非工兵不可攻击地雷
        st4 = mk({(0, 0): ("r", "PAI", True), (0, 1): ("b", "LEI", True)})
        self.assertNotIn((0, 1), dests(st4, (0, 0)))
        # 工兵挖雷合法
        st5 = mk({(0, 0): ("r", "GONG", True), (0, 1): ("b", "LEI", True)})
        self.assertIn((0, 1), dests(st5, (0, 0)))
        # 旧规则开关可恢复自杀走法
        st6 = mk({(0, 0): ("r", "PAI", True), (0, 1): ("b", "SI", True)},
                 cfg=RuleConfig(allow_suicide_attack=True))
        self.assertIn((0, 1), dests(st6, (0, 0)))

    def test_engineer_flight_with_detour(self):
        # 工兵全场任意转弯：纵向铁路只到 row 1..10（不到达底线 row 0/11）
        st = mk({(5, 0): ("r", "GONG", True)})
        d = dests(st, (5, 0))
        for r in range(1, 11):             # col0 纵线 row 1..10 可达
            if r == 5:
                continue                   # 起点自身不是落点
            self.assertIn((r, 0), d, (r, 0))
        self.assertNotIn((0, 0), d)        # row 0 非铁路
        self.assertNotIn((11, 0), d)       # row 11 非铁路
        for c in range(5):                 # row5 / row6 横线可达
            if c != 0:
                self.assertIn((5, c), d, (5, c))
            self.assertIn((6, c), d, (6, c))
        # 经前线 (5,4) 转弯进入 col4 纵线
        self.assertIn((7, 4), d)
        self.assertIn((10, 4), d)
        # 倒数第二排横向铁路可达
        for c in (1, 2, 3):
            self.assertIn((1, c), d, (1, c))
            self.assertIn((10, c), d, (10, c))

    def test_engineer_on_bottom_line_turns(self):
        # 工兵在倒数第二排（row1/10）也可转弯
        st = mk({(1, 2): ("r", "GONG", True)})
        d = dests(st, (1, 2))
        self.assertIn((1, 0), d)
        self.assertIn((1, 4), d)
        self.assertIn((0, 2), d)           # 公路一步正常
        self.assertNotIn((0, 0), d)        # (0,0) 非铁路，不可由 (1,2) 铁路滑入
        # 开关关闭时工兵退化为直线滑行
        st2 = mk({(5, 0): ("r", "GONG", True)},
                 cfg=RuleConfig(engineer_rail_turns=False))
        d2 = dests(st2, (5, 0))
        self.assertIn((5, 2), d2)          # row5 直线仍可达
        self.assertNotIn((6, 2), d2)       # 不能拐进 row6

    def test_engineer_cannot_fly_over_pieces(self):
        # 修正规则：工兵不可越过轨道上的棋子移动（将左右与中桥三方堵住）
        st = mk({(5, 2): ("r", "GONG", True),
                 (5, 1): ("b", "SI", True), (5, 3): ("b", "SI", True),
                 (6, 2): ("b", "SI", True)})
        d = dests(st, (5, 2))
        self.assertNotIn((5, 1), d)        # 敌司令禁自杀不可攻击
        self.assertNotIn((5, 3), d)
        self.assertNotIn((6, 2), d)        # 中桥敌司令阻挡
        self.assertNotIn((5, 0), d)        # 被 (5,1) 阻挡，不可越过
        self.assertNotIn((5, 4), d)        # 被 (5,3) 阻挡，不可越过
        # 公路一步仍可用
        self.assertIn((4, 2), d)

    def test_engineer_rail_capture_and_blocked(self):
        # (2,0) 红工兵被 (1,0) 和 (3,0) 敌工兵夹在纵向轨道中，可吃 (1,0) 和 (3,0)，但不可越过它们去往其他铁路线
        st = mk({(2, 0): ("r", "GONG", True),
                 (1, 0): ("b", "GONG", True),
                 (3, 0): ("b", "GONG", True)})
        d = dests(st, (2, 0))
        self.assertIn((1, 0), d)           # 可吃 (1,0)
        self.assertIn((3, 0), d)           # 可吃 (3,0)
        self.assertNotIn((4, 0), d)        # 路径被 (3,0) 阻挡，不可越过到达 (4,0)
        self.assertNotIn((5, 0), d)        # 不可越过到达 (5,0)

    def test_flag_gate_needs_mines_cleared(self):
        dead = tuple(Piece("b", Rank.LEI, True) for _ in range(3))
        # 未清雷：工兵也不能攻击军旗
        st = mk({(0, 0): ("r", "GONG", True), (0, 1): ("b", "QI", True)})
        self.assertNotIn((0, 1), dests(st, (0, 0)))
        # 清完 3 雷：工兵可以扛旗（APK：军旗只有工兵能吃）
        st2 = mk({(0, 0): ("r", "GONG", True), (0, 1): ("b", "QI", True)}, dead=dead)
        self.assertIn((0, 1), dests(st2, (0, 0)))
        # 清雷后扛旗 → 获胜
        nxt = st2.apply(Action("move", (0, 0), (0, 1)))
        self.assertEqual(nxt.winner, 0)
        self.assertEqual(nxt.win_reason, "flag")

    def test_flag_gong_only(self):
        # APK 翻棋规则：军旗只有工兵能吃——排长/司令/炸弹清雷后也不可扛旗
        dead = tuple(Piece("b", Rank.LEI, True) for _ in range(3))
        for rank in ("PAI", "SI", "ZHA"):
            st = mk({(0, 0): ("r", rank, True), (0, 1): ("b", "QI", True)},
                    dead=dead)
            self.assertNotIn((0, 1), dests(st, (0, 0)), rank)
        # 旧规则（App"吃军旗:任何棋子"档）：炸弹可扛旗并终局
        st = mk({(0, 0): ("r", "ZHA", True), (0, 1): ("b", "QI", True)},
                dead=dead, cfg=RuleConfig(flag_gong_only=False))
        self.assertIn((0, 1), dests(st, (0, 0)))
        nxt = st.apply(Action("move", (0, 0), (0, 1)))
        self.assertEqual(nxt.winner, 0)

    def test_engineer_digs_mine(self):
        st = mk({(0, 0): ("r", "GONG", True), (0, 1): ("b", "LEI", True)})
        nxt = st.apply(Action("move", (0, 0), (0, 1)))
        self.assertEqual(nxt.board[(0, 1)], Piece("r", Rank.GONG, True))
        self.assertEqual(len(nxt.dead), 1)

    def test_attacker_dies_on_mine(self):
        # 旧规则（允许自杀）下碰雷阵亡的结算仍需正确
        st = mk({(0, 0): ("r", "SI", True), (0, 1): ("b", "LEI", True)},
                cfg=RuleConfig(allow_suicide_attack=True))
        nxt = st.apply(Action("move", (0, 0), (0, 1)))
        self.assertNotIn((0, 0), nxt.board)
        self.assertEqual(nxt.board[(0, 1)], Piece("b", Rank.LEI, True))
        self.assertEqual(len(nxt.dead), 1)

    def test_equal_ranks_both_die(self):
        st = mk({(0, 0): ("r", "LV", True), (0, 1): ("b", "LV", True)})
        nxt = st.apply(Action("move", (0, 0), (0, 1)))
        self.assertNotIn((0, 1), nxt.board)
        self.assertEqual(len(nxt.dead), 2)


class TestGameEnd(unittest.TestCase):
    def test_immobilized(self):
        # 红方走完后，蓝方只剩地雷+军旗 → 无子可动判负
        st = mk({(5, 2): ("r", "PAI", True), (11, 1): ("b", "LEI", True),
                 (11, 3): ("b", "QI", True)}, turn=0)
        self.assertTrue(st.legal_actions())
        nxt = st.apply(Action("move", (5, 2), (5, 3)))
        self.assertEqual(nxt.winner, 0)
        self.assertEqual(nxt.win_reason, "immobilized")

    def test_draw_at_max_plies(self):
        cfg = RuleConfig(max_plies=4, no_capture_draw_plies=0)  # 关闭70步线，单测手数线
        st = mk({(5, 2): ("r", "PAI", True), (10, 1): ("b", "PAI", True),
                 (11, 3): ("b", "QI", True)}, turn=0, cfg=cfg)
        cur = st
        back_and_forth = [Action("move", (5, 2), (5, 1)),
                          Action("move", (10, 1), (10, 2)),
                          Action("move", (5, 1), (5, 2)),
                          Action("move", (10, 2), (10, 1))]
        for i, act in enumerate(back_and_forth):
            cur = cur.apply(act)
        self.assertEqual(cur.winner, -1)
        self.assertEqual(cur.win_reason, "max_plies")

    def test_no_capture_draw_and_reset(self):
        # APK：连续 70 步未吃子判和——这里用阈值 3 快速验证计数、判和与吃子清零
        cfg = RuleConfig(no_capture_draw_plies=3, max_plies=1000)
        st = mk({(5, 2): ("r", "PAI", True), (10, 1): ("b", "PAI", True),
                 (11, 3): ("b", "QI", True)}, turn=0, cfg=cfg)
        cur = st
        for act in [Action("move", (5, 2), (5, 1)),
                    Action("move", (10, 1), (10, 2))]:
            cur = cur.apply(act)
        self.assertIsNone(cur.winner)
        self.assertEqual(cur.quiet, 2)
        # 第 3 手无吃子 → 判和
        cur = cur.apply(Action("move", (5, 1), (5, 2)))
        self.assertEqual(cur.winner, -1)
        self.assertEqual(cur.win_reason, "no_capture")
        # 吃子清零计数：排长吃工兵后 quiet 归 0，不触发和棋
        # （蓝方另留一个远处活动子，避免"无棋可走"先触发判负）
        st2 = mk({(5, 2): ("r", "PAI", True), (5, 3): ("b", "GONG", True),
                  (10, 1): ("b", "PAI", True), (11, 3): ("b", "QI", True)},
                 turn=0, cfg=cfg)
        cur2 = st2.apply(Action("move", (5, 2), (5, 3)))   # 排长吃工兵
        self.assertEqual(cur2.quiet, 0)
        self.assertIsNone(cur2.winner)

    def test_position_key_observable(self):
        # 循环判和按"可观察局面"：暗子身份不同但明暗分布相同 → 同一键
        st = deal(random.Random(7))
        k1 = position_key(st)
        w = st.sample_world(random.Random(1))
        st2 = st.instantiate(w)          # 暗子身份被"赋值"，可观察面不变
        self.assertEqual(k1, position_key(st2))
        st3 = st.apply(Action("flip", st.hidden_positions()[0]))
        self.assertNotEqual(k1, position_key(st3))

    def test_sample_world_consistency(self):
        st = deal(random.Random(99))
        from collections import Counter
        expect = Counter((c, r) for c in ("r", "b") for r, n in COMPOSITION.items()
                         for _ in range(n))
        w = st.sample_world(random.Random(5))
        got = Counter((pc.color, pc.rank) for pc in w.values())
        self.assertEqual(got, expect)
        m = st.marginal(st.hidden_positions()[0])
        self.assertAlmostEqual(sum(m.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
