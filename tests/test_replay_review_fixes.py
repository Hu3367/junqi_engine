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

        engine = HybridDecisionEngine(k_worlds=2, seed=42)
        scored = engine.choose_actions(st, topn=len(st.legal_actions()))
        score_dict = {a: s for a, s in scored}

        # 1. 军长吃排长 (1, 1) -> (1, 2)
        act_jun = Action('move', (1, 1), (1, 2))
        self.assertIn(act_jun, score_dict)
        self.assertGreater(score_dict[act_jun], 50.0, '军长吃排长必须获得显著的胜势吃子保护分！')

        # 2. 营长吃排长 (2, 2) -> (1, 2)
        act_ying = Action('move', (2, 2), (1, 2))
        self.assertIn(act_ying, score_dict)
        self.assertGreater(score_dict[act_ying], 50.0, '营长吃排长必须获得显著的胜势吃子保护分！')

        # 3. 工兵挖地雷 (3, 3) -> (3, 4)
        act_gong = Action('move', (3, 3), (3, 4))
        self.assertIn(act_gong, score_dict)
        self.assertGreater(score_dict[act_gong], 80.0, '工兵挖地雷是关键胜势吃子，必须获得极高加分！')

        # 4. 炸弹兑司令 (1, 0) -> (2, 0)
        act_bomb = Action('move', (1, 0), (2, 0))
        self.assertIn(act_bomb, score_dict)
        self.assertGreater(score_dict[act_bomb], 50.0, '炸弹兑掉司令具有极高战略价值，必须获得战术加分！')

        # 5. 验证自杀拦截规则 (_apply_tactical_rules 对 defender_wins 的严惩)
        fake_suicide_act = Action('move', (1, 2), (2, 4)) # 蓝排长撞蓝司令(若被作为攻击测试)
        # 用带自杀的模拟动作测试 _apply_tactical_rules
        test_board = {
            (5, 0): Piece('r', Rank.PAI, revealed=True),
            (6, 0): Piece('b', Rank.JUN, revealed=True),
        }
        test_st = GameState(board=test_board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                            first_flip_done=True, ply=20, winner=None, win_reason=None, cfg=self.cfg)
        suicide_act = Action('move', (5, 0), (6, 0))
        tactical_res = engine._apply_tactical_rules(test_st, {suicide_act: 0.5})
        self.assertLess(tactical_res[0][1], -100.0, '自杀性吃子必须被扣除大分！')

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
        若红工兵贪吃地雷出营，下一手立即被蓝连长反杀并丢营。
        验证战术规则严惩弃营吃小子，优先守营或营间机动。"""
        board = {
            (7, 3): Piece('r', Rank.GONG, revealed=True),  # 红工兵在行营
            (6, 4): Piece('b', Rank.LEI, revealed=True),   # 蓝地雷在铁路兵站
            (6, 3): Piece('b', Rank.LIAN, revealed=True),  # 蓝连长在兵站（可反杀 6,4 且可入驻 7,3）
            (5, 2): Piece('b', Rank.PAI, revealed=False),  # 暗子
        }
        st = GameState(board=board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                       first_flip_done=True, ply=8, winner=None, win_reason=None, cfg=self.cfg)

        engine = HybridDecisionEngine(k_worlds=2, seed=42)
        scored = engine.choose_actions(st, topn=len(st.legal_actions()))
        score_dict = {a: s for a, s in scored}

        act_abandon = Action('move', (7, 3), (6, 4))
        self.assertIn(act_abandon, score_dict)
        # 验证弃营吃小子且面临反杀和丢营的动作被重重扣分
        self.assertLess(score_dict[act_abandon], -200.0,
                        f'弃营挖雷遭连长反杀并丢营，必须受到严厉扣分，实际得分: {score_dict[act_abandon]}')

        # 验证最优动作为翻暗子或营内走步，绝不是弃营吃雷
        best_act = scored[0][0]
        self.assertNotEqual(best_act, act_abandon, 'AI绝不能选择弃营自杀性吃小子！')

    def test_bomb_suicide_on_minor_penalized(self):
        """验证炸弹主动撞廉价小子 (排长/工兵等) 自爆被扣除大分，严禁贱卖战略重器。"""
        board = {
            (5, 0): Piece('r', Rank.ZHA, revealed=True),  # 红炸弹在铁路
            (6, 0): Piece('b', Rank.PAI, revealed=True),  # 蓝排长在铁路
            (0, 0): Piece('b', Rank.SI, revealed=False),  # 暗子
        }
        st = GameState(board=board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                       first_flip_done=True, ply=20, winner=None, win_reason=None, cfg=self.cfg)

        engine = HybridDecisionEngine(k_worlds=2, seed=42)
        scored = engine.choose_actions(st, topn=len(st.legal_actions()))
        score_dict = {a: s for a, s in scored}

        act_bomb_pai = Action('move', (5, 0), (6, 0))
        self.assertIn(act_bomb_pai, score_dict)
        self.assertLess(score_dict[act_bomb_pai], -200.0,
                        f'炸弹撞排长自爆必须被严重扣分，实际得分: {score_dict[act_bomb_pai]}')

    def test_camp_adjacent_flip_bonus(self):
        """验证邻接己方已占领行营的翻棋动作获得 +15.0 战术加分，破除龟缩拒翻。"""
        # 红方占领行营 (7, 3)，邻接暗子为 (6, 3) 与 (8, 3)；远离行营暗子为 (0, 0)
        board = {
            (7, 3): Piece('r', Rank.GONG, revealed=True),  # 红工兵在行营
            (6, 3): Piece('b', Rank.PAI, revealed=False),  # 邻营暗子
            (0, 0): Piece('b', Rank.PAI, revealed=False),  # 远端暗子
        }
        st = GameState(board=board, dead=[], seat_color={0: 'r', 1: 'b'}, turn=0,
                       first_flip_done=True, ply=20, winner=None, win_reason=None, cfg=self.cfg)

        engine = HybridDecisionEngine(k_worlds=2, seed=42)
        scored = engine.choose_actions(st, topn=len(st.legal_actions()))
        score_dict = {a: s for a, s in scored}

        act_adj_flip = Action('flip', (6, 3))
        act_far_flip = Action('flip', (0, 0))
        self.assertIn(act_adj_flip, score_dict)
        self.assertIn(act_far_flip, score_dict)
        # 邻营翻棋加分应显著高于远端翻棋
        self.assertGreater(score_dict[act_adj_flip], score_dict[act_far_flip] + 10.0,
                           f'邻营翻棋得分 ({score_dict[act_adj_flip]}) 应显著高于远端翻棋 ({score_dict[act_far_flip]})')


if __name__ == '__main__':
    unittest.main()