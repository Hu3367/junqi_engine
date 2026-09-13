"""复盘点评战术修正回归测试（P2 重构后契约）。

2026-09-13 hybrid_engine 定价重构后，"好与坏"由 QSearch 战术搜索按真实
子力/战术后果定价，先验仅做有界平局打破（prior_band=15 分）。本文件保留
原复盘点评的四条棋理，断言更新为搜索定价语义：
1. 胜势吃子永不被惩罚（得分为正），且兑取高价值目标的动作排序最高；
2. 防守方司令沿铁路存在两步反杀线时，军长吃排长的净收益被搜索诚实折价
   （旧契约的固定 +50 分已废除）；
3. 弃营挖雷遭"连长进营 -> 行营扑杀特权反杀"（深度 3 可见的真实反驳线）
   被定价为净负，绝非最优动作；
4. 炸弹撞廉价小子自爆为净负分且非最优（严禁贱卖战略重器）；
5. 自杀性攻击（defender_wins）在规则层被硬动作掩码拦截，根本不合法。
"""
import unittest
from junqi.state import GameState, Piece, Action
from junqi.rules import Rank
from junqi.config import RuleConfig, EvalWeights
from junqi.hybrid_engine import HybridDecisionEngine
from junqi.search import ExpertSearchEngine


class TestReplayReviewFixes(unittest.TestCase):
    def setUp(self):
        self.cfg = RuleConfig()
        self.weights = EvalWeights()

    def test_winning_captures_not_penalized(self):
        # 所有吃子目标必须处于兵站（非行营），且双方处于合法走步邻接
        board = {
            (1, 1): Piece('r', Rank.JUN, revealed=True),   # 红军长
            (2, 2): Piece('r', Rank.YING, revealed=True),  # 红营长
            (3, 3): Piece('r', Rank.GONG, revealed=True),  # 红工兵
            (1, 0): Piece('r', Rank.ZHA, revealed=True),   # 红炸弹 (西侧铁路)
            (1, 2): Piece('b', Rank.PAI, revealed=True),   # 蓝排长 (兵站，可被军长或营长吃)
            (3, 4): Piece('b', Rank.LEI, revealed=True),   # 蓝地雷 (东侧铁路兵站，可被工兵挖)
            (2, 0): Piece('b', Rank.SI, revealed=True),    # 蓝司令 (西侧铁路兵站，可被炸弹兑)
        }
        st = GameState(board=board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                       first_flip_done=True, ply=20, winner=None, win_reason=None, cfg=self.cfg)

        engine = HybridDecisionEngine(k_worlds=2, seed=42, candidate_k=64)
        scored = engine.choose_actions(st, topn=len(st.legal_actions()))
        score_dict = {a: s for a, s in scored}

        # 1. 军长吃排长 (1, 1) -> (1, 2)：得分为正（不吃亏）。
        # 注意：蓝司令沿铁路两步内可反杀落点军长，搜索诚实折价
        # （实测约 +4，旧契约的固定 +50 分属量纲崩溃产物，已废除）
        act_jun = Action('move', (1, 1), (1, 2))
        self.assertIn(act_jun, score_dict)
        self.assertGreater(score_dict[act_jun], 0.0,
                           '军长吃排长（净 +18 子力）必须为正分！')

        # 2. 营长吃排长 (2, 2) -> (1, 2)：得分为正，且优于军长版
        # （军长压上后被司令两步反杀的暴露度更高，搜索定价更低）
        act_ying = Action('move', (2, 2), (1, 2))
        self.assertIn(act_ying, score_dict)
        self.assertGreater(score_dict[act_ying], 0.0,
                           '营长吃排长（净 +18 子力）必须为正分！')
        self.assertGreater(score_dict[act_ying], score_dict[act_jun],
                           '营长吃排长的暴露代价应低于军长吃排长')

        # 3. 工兵挖地雷 (3, 3) -> (3, 4)：得分为正（净 +30 子力）
        act_gong = Action('move', (3, 3), (3, 4))
        self.assertIn(act_gong, score_dict)
        self.assertGreater(score_dict[act_gong], 0.0, '工兵挖地雷必须为正分！')

        # 4. 炸弹兑司令 (1, 0) -> (2, 0)：净 +48 子力，全场最优动作
        act_bomb = Action('move', (1, 0), (2, 0))
        self.assertIn(act_bomb, score_dict)
        self.assertGreater(score_dict[act_bomb], 30.0,
                           '炸弹兑掉司令（净 +48 子力）必须是最高收益动作！')
        best_act = scored[0][0]
        self.assertEqual(best_act, act_bomb, '炸弹兑司令应排名全场第一')

        # 5. 自杀拦截（原 _apply_tactical_rules 的 defender_wins 严惩）：
        # 重构后由规则层硬动作掩码承担——defender_wins 的攻击根本不合法
        test_board = {
            (5, 0): Piece('r', Rank.PAI, revealed=True),
            (6, 0): Piece('b', Rank.JUN, revealed=True),
        }
        test_st = GameState(board=test_board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                            first_flip_done=True, ply=20, winner=None, win_reason=None, cfg=self.cfg)
        suicide_act = Action('move', (5, 0), (6, 0))
        self.assertNotIn(suicide_act, test_st.legal_actions(),
                         '排长撞军长（defender_wins）必须被硬动作掩码拦截！')

    def test_policy_invariance_across_sampled_worlds(self):
        board = {
            (1, 1): Piece('r', Rank.JUN, revealed=True),
            (2, 2): Piece('b', Rank.SHI, revealed=True),
            (0, 0): Piece('r', Rank.PAI, revealed=False),
            (0, 4): Piece('b', Rank.GONG, revealed=False),
        }
        st = GameState(board=board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                       first_flip_done=True, ply=10, winner=None, win_reason=None, cfg=self.cfg)

        engine1 = HybridDecisionEngine(k_worlds=4, seed=100)
        engine2 = HybridDecisionEngine(k_worlds=4, seed=999)

        res1 = engine1.evaluate_position(st)
        res2 = engine2.evaluate_position(st)

        acts1 = dict(res1['action_scores'])
        acts2 = dict(res2['action_scores'])

        self.assertEqual(len(acts1), len(acts2))
        for a in acts1:
            self.assertAlmostEqual(acts1[a], acts2[a], places=5,
                                   msg=f'公共 Policy 输出在不同采样世界下必须绝对不变！动作 {a} 发生漂移')

    def test_chance_node_terminal_handling_on_flip(self):
        search_eng = ExpertSearchEngine(weights=self.weights, seed=42)

        board = {
            (10, 4): Piece('r', Rank.SI, revealed=True),
            (11, 3): Piece('r', Rank.JUN, revealed=True),
            (11, 4): Piece('b', Rank.PAI, revealed=False),
        }
        st = GameState(board=board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                       first_flip_done=True, ply=50, winner=None, win_reason=None, cfg=self.cfg)

        flip_act = Action('flip', (11, 4))
        score = search_eng._evaluate_chance_flip(st, flip_act, depth=1, ply_depth=0,
                                                 alpha=-10000.0, beta=10000.0, path_history=set())
        self.assertGreater(score, 5000.0, '翻开暗子造成对手困毙终局时，几率节点必须回传终局胜势高分！')

    def test_camp_preservation_over_minor_piece_bait(self):
        """复现 game_20260906_192146 第 8 手：
        红工兵在行营 (7, 3)，蓝地雷在 (6, 4)，蓝连长在 (6, 3)。
        弃营挖雷的真实反驳线（深度 3 可见）：连长入驻空营 (7, 3) 后以行营
        扑杀特权吃掉 (6, 4) 的工兵——净亏 42-30 且丢营。
        验证搜索定价下该动作是净负分且绝非最优。"""
        board = {
            (7, 3): Piece('r', Rank.GONG, revealed=True),  # 红工兵在行营
            (6, 4): Piece('b', Rank.LEI, revealed=True),   # 蓝地雷在铁路兵站
            (6, 3): Piece('b', Rank.LIAN, revealed=True),  # 蓝连长在兵站（可入驻空营后扑杀）
            (5, 2): Piece('b', Rank.PAI, revealed=False),  # 暗子
        }
        st = GameState(board=board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                       first_flip_done=True, ply=8, winner=None, win_reason=None, cfg=self.cfg)

        # 反驳线深 3 ply（挖雷 -> 连长进营 -> 行营扑杀），需 tactical_depth=3
        engine = HybridDecisionEngine(k_worlds=2, seed=42, candidate_k=64,
                                      tactical_depth=3)
        scored = engine.choose_actions(st, topn=len(st.legal_actions()))
        score_dict = {a: s for a, s in scored}

        act_abandon = Action('move', (7, 3), (6, 4))
        self.assertIn(act_abandon, score_dict)
        # 弃营挖雷被"连长进营 -> 扑杀"反驳：净负分且绝非最优
        # （旧契约的绝对分值 < -200 属量纲崩溃产物，已废除）
        self.assertLess(score_dict[act_abandon], 0.0,
                        f'弃营挖雷遭进营扑杀反驳必须为净负分，实际得分: {score_dict[act_abandon]}')
        best_act = scored[0][0]
        self.assertNotEqual(best_act, act_abandon, 'AI绝不能选择弃营挖雷送工兵！')

    def test_bomb_suicide_on_minor_penalized(self):
        """验证炸弹主动撞廉价小子 (排长) 自爆为净负分且非最优，严禁贱卖战略重器。
        （净子力 52-18 = -34，远端另置红排长避免困毙终局污染定价）"""
        board = {
            (5, 0): Piece('r', Rank.ZHA, revealed=True),  # 红炸弹在铁路
            (6, 0): Piece('b', Rank.PAI, revealed=True),  # 蓝排长在铁路
            (10, 1): Piece('r', Rank.PAI, revealed=True),  # 红排长远处机动子
            (1, 1): Piece('b', Rank.PAI, revealed=True),   # 蓝排长远处机动子
        }
        st = GameState(board=board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                       first_flip_done=True, ply=20, winner=None, win_reason=None, cfg=self.cfg)

        engine = HybridDecisionEngine(k_worlds=2, seed=42, candidate_k=64)
        scored = engine.choose_actions(st, topn=len(st.legal_actions()))
        score_dict = {a: s for a, s in scored}

        act_bomb_pai = Action('move', (5, 0), (6, 0))
        self.assertIn(act_bomb_pai, score_dict)
        self.assertLess(score_dict[act_bomb_pai], 0.0,
                        f'炸弹撞排长自爆（净 -34 子力）必须为净负分，实际得分: {score_dict[act_bomb_pai]}')
        best_act = scored[0][0]
        self.assertNotEqual(best_act, act_bomb_pai, 'AI绝不能选择炸弹自爆贱卖！')

    def test_camp_adjacent_flip_bonus(self):
        """验证邻接己方行营的翻棋定价高于远端翻棋（依托据点辐射拓荒棋理，
        实证 96.2% 邻营翻棋）。旧契约的固定 +15 加分已废除——量级差异现在
        由有界先验（BC 策略从 9.5 万人类 ply 学到的邻营翻棋偏好）与
        搜索对翻子后行营辐射区的估值共同承载。"""
        # 红方占领行营 (7, 3)，邻接暗子为 (6, 3)；远离行营暗子为 (0, 0)
        board = {
            (7, 3): Piece('r', Rank.GONG, revealed=True),  # 红工兵在行营
            (6, 3): Piece('b', Rank.PAI, revealed=False),  # 邻营暗子
            (0, 0): Piece('b', Rank.PAI, revealed=False),  # 远端暗子（冻结位）
        }
        st = GameState(board=board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                       first_flip_done=True, ply=20, winner=None, win_reason=None, cfg=self.cfg)

        engine = HybridDecisionEngine(k_worlds=2, seed=42, candidate_k=64)
        scored = engine.choose_actions(st, topn=len(st.legal_actions()))
        score_dict = {a: s for a, s in scored}

        act_adj_flip = Action('flip', (6, 3))
        act_far_flip = Action('flip', (0, 0))
        self.assertIn(act_adj_flip, score_dict)
        self.assertIn(act_far_flip, score_dict)
        # 邻营翻棋定价高于远端翻棋（方向性断言；远端 (0,0) 为冻结位，
        # 翻出的子无法进营，搜索估值同样反映这一差异）
        self.assertGreater(score_dict[act_adj_flip], score_dict[act_far_flip],
                           f'邻营翻棋得分 ({score_dict[act_adj_flip]:.2f}) 应高于远端翻棋 ({score_dict[act_far_flip]:.2f})')


if __name__ == '__main__':
    unittest.main()
