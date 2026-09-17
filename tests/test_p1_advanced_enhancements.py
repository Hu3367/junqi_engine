"""P1 高阶增强与 IDS 限时控制专项测试套件。

涵盖：
1. IDS 1000ms 动态时间预算与深度探索；
2. P1.A: 内层翻棋关键区域解禁搜索（受限 Star1）；
3. P1.B: fit_weights 严格公共信息边界（零暗子身份泄漏）；
4. P1.C: QSearch 高危大子受威胁逃入行营候选；
5. P1.D: C++ 驱动 HybridAgent 混合引擎集成。
"""
import random
import pytest

from junqi.state import GameState, Piece, Action
from junqi.rules import Rank, battle
from junqi.config import RuleConfig, SearchConfig, EvalWeights
from junqi.core_bridge import HAS_CPP_CORE
from junqi.search import (ExpertSearchEngine, _STAR1_IMPORTANCE,
                          _STAR1_KIND_ORDER)
from junqi.ai import ExpertAgent, HybridAgent
from junqi.fit_weights import (FEATURE_NAMES, _project, default_vector, features,
                               to_eval_weights)
from junqi.selfplay import deal


class TestIDSConfigAndExecution:
    """测试 0：IDS 1000ms 配置与加深探索"""

    def test_default_search_config_has_1000ms(self):
        cfg = SearchConfig()
        assert cfg.time_limit_ms == 1000
        # `depth` 同时是迭代加深的深度上限 ⇒ 不需要 `ids_max_depth`
        # （它曾是死配置：声明后无任何消费者，已删除）
        assert cfg.depth == 2
        assert not hasattr(cfg, "ids_max_depth")

    def test_expert_agent_ids_execution(self):
        st = deal(random.Random(42), RuleConfig())
        agent = ExpertAgent(seed=42)
        assert agent.cfg.time_limit_ms == 1000
        assert agent.cfg.depth == 2
        act = agent.select_action(st)
        assert act is not None
        assert agent.engine.stats.max_depth == 2

        # 显式加深到 depth=4
        deeper_agent = ExpertAgent(search=SearchConfig(depth=4, time_limit_ms=1000), seed=42)
        act4 = deeper_agent.select_action(st)
        assert act4 is not None
        assert deeper_agent.engine.stats.max_depth >= 4


class TestP1ARestrictedStar1:
    """测试 P1.A：受限 Star1 内层交火翻棋展开。

    断言口径（用**首次发牌**局面：全盘皆暗、无已明子邻接）：
      · 静区翻棋（第 0/11 行，不邻接任何行营）⇒ 仍走解析期望，取值与 depth 无关；
      · 战术区翻棋（邻接行营）⇒ depth>=3 时解禁受限展开，取值随 depth 变化。
    这同时锁住"门槛确实按位置生效"，而不是只断言返回类型（旧版正是如此，形同虚设）。
    """

    QUIET = ((0, 0), (0, 4), (11, 2))
    TACTICAL = ((3, 1), (3, 3), (4, 2))

    def test_quiet_zone_flip_stays_depth_invariant(self):
        st = deal(random.Random(7), RuleConfig())
        eng = ExpertSearchEngine(seed=7, use_cpp_search=False)
        for pos in self.QUIET:
            v2 = eng._evaluate_chance_flip(st, Action("flip", pos), 2, 1,
                                           -999999.0, 999999.0, set())
            v3 = eng._evaluate_chance_flip(st, Action("flip", pos), 3, 1,
                                           -999999.0, 999999.0, set())
            assert v2 == pytest.approx(v3, abs=1e-9), \
                f"静区 {pos} 不应展开受限 Star1: d2={v2} d3={v3}"

    def test_tactical_zone_flip_is_expanded_at_depth3(self):
        st = deal(random.Random(7), RuleConfig())
        eng = ExpertSearchEngine(seed=7, use_cpp_search=False)
        for pos in self.TACTICAL:
            v2 = eng._evaluate_chance_flip(st, Action("flip", pos), 2, 1,
                                           -999999.0, 999999.0, set())
            v3 = eng._evaluate_chance_flip(st, Action("flip", pos), 3, 1,
                                           -999999.0, 999999.0, set())
            assert abs(v3 - v2) > 1e-6, \
                f"战术区 {pos} 在 depth>=3 应展开受限 Star1: d2={v2} d3={v3}"


class TestStar1ImportanceTable:
    """受限展开的"战术重要性"分档表必须两侧一致。

    背景：受限展开原取"剩余数量降序前 3"，命中连长/排长/工兵/地雷（每种 3 枚），
    而司令/军长/炸弹（1~2 枚）总落入长尾 ⇒ 与"消除翻出大子被吃盲区"目标相反。
    """

    def test_python_table_prioritises_decisive_ranks(self):
        imp = _STAR1_IMPORTANCE
        for decisive in (Rank.ZHA, Rank.SI, Rank.JUN):
            for bulk in (Rank.LIAN, Rank.PAI, Rank.GONG, Rank.LEI):
                assert imp[decisive] > imp[bulk], \
                    f"{decisive.name} 的战术重要性应高于 {bulk.name}"

    def test_cpp_and_python_tables_match(self):
        if not HAS_CPP_CORE:
            pytest.skip("junqi_core 未编译")
        import junqi_core as jc
        py = [_STAR1_IMPORTANCE[r] for r in _STAR1_KIND_ORDER]
        assert jc.expert_star1_importance() == py, \
            "C++ 的 STAR1_IMPORTANCE 与 Python `_STAR1_IMPORTANCE` 漂移"


class TestP1BPublicInformationBoundary:
    """测试 P1.B：fit_weights 严格公共信息边界（无暗子窥探）"""

    def test_unflipped_board_has_all_zero_features(self):
        st = deal(random.Random(100), RuleConfig())
        f0 = features(st, 0)
        f1 = features(st, 1)
        assert all(v == 0.0 for v in f0.values())
        assert all(v == 0.0 for v in f1.values())

    def test_changing_hidden_piece_rank_does_not_leak(self):
        st = deal(random.Random(200), RuleConfig())
        flip_pos = next(p for p, pc in st.board.items() if not pc.revealed)
        st = st.apply(Action("flip", flip_pos))
        assert st.first_flip_done

        f_before = features(st, 0)

        hidden_positions = [p for p, pc in st.board.items() if not pc.revealed]
        p1, p2 = hidden_positions[0], hidden_positions[1]
        pc1, pc2 = st.board[p1], st.board[p2]
        if pc1.rank != pc2.rank or pc1.color != pc2.color:
            st.board[p1] = Piece(pc2.color, pc2.rank, revealed=False)
            st.board[p2] = Piece(pc1.color, pc1.rank, revealed=False)

            f_after = features(st, 0)
            for k in FEATURE_NAMES:
                assert f_before[k] == pytest.approx(f_after[k], abs=1e-6), \
                    f"Feature '{k}' leaked hidden piece identity! before={f_before[k]}, after={f_after[k]}"

    def test_weight_vector_roundtrip(self):
        vec = default_vector()
        assert len(vec) == len(FEATURE_NAMES)
        w = to_eval_weights(vec)
        assert isinstance(w, EvalWeights)
        assert w.camp_occ == 10.0

    def test_roundtrip_default_vector_reproduces_eval_weights(self):
        """`to_eval_weights(default_vector())` 必须逐字段等于 `EvalWeights()`。"""
        assert to_eval_weights(default_vector()) == EvalWeights()

    def test_projection_is_fixed_point_of_warm_start(self):
        """`_project(default_vector())` 必须等于 `default_vector()`。

        回归：`_bounds()` 曾用**写死下标**表达语义约束，`FEATURE_NAMES` 从 14 维扩到
        25 维后约束落到错误特征上 —— 实测一次投影就把 flag_exposed 40→2、
        camp_siege 3→0、hq_locked −8→0（符号反转）、mine_guard 8→2，
        而 threat/attack/attack_camp 变成完全无界。现改为按特征名索引。
        """
        init = default_vector()
        assert _project(list(init)) == init

    def test_every_fitted_feature_maps_back_to_eval_weights(self):
        """每个特征都必须能写回 EvalWeights，唯一例外是 flip_bias（分析用）。

        回归：`mobility` 曾被列入 FEATURE_NAMES，但 `EvalWeights` 没有对应字段，
        拟合出的系数会被静默丢弃（且 evaluate_expert 的机动力项是另一套定义 +
        硬编码 0.5，系数无法迁移）⇒ 已从特征集移除。
        """
        base = default_vector()
        w0 = to_eval_weights(list(base))
        inert = []
        for i, name in enumerate(FEATURE_NAMES):
            probe = list(base)
            probe[i] += 5.0
            if to_eval_weights(probe) == w0:
                inert.append(name)
        assert inert == ["flip_bias"], f"这些特征拟合后无法写回 EvalWeights：{inert}"

    def test_features_keys_match_feature_names_exactly(self):
        st = deal(random.Random(17), RuleConfig())
        for seat in (0, 1):
            assert set(features(st, seat)) == set(FEATURE_NAMES)


class TestP1CQSearchRetreatIntoCamp:
    """测试 P1.C：QSearch 高危大子逃入行营避难走法"""

    def test_threatened_major_retreats_into_camp(self):
        st = GameState(board={
            (3, 1): Piece("r", Rank.SHI, revealed=True),
            (4, 1): Piece("b", Rank.JUN, revealed=True),
            (0, 1): Piece("b", Rank.QI, revealed=True),
            (11, 1): Piece("r", Rank.QI, revealed=True),
        }, turn=0, cfg=RuleConfig())
        st.seat_color[0], st.seat_color[1] = "r", "b"
        st.first_flip_done = True

        eng = ExpertSearchEngine(seed=7, use_cpp_qsearch=True, use_cpp_search=True)
        act, score, _ = eng.search(st, max_depth=1)
        assert act == Action("move", (3, 1), (3, 2))


class TestP1DHybridAgentIntegration:
    """测试 P1.D：C++ 驱动 HybridAgent 混合引擎集成"""

    def test_hybrid_agent_initialization_and_tactical_override(self):
        agent = HybridAgent(search_depth=3, top_k=4, seed=42)
        assert agent.search_depth == 3
        assert agent.engine.use_cpp_search

        st = GameState(board={
            (1, 1): Piece("r", Rank.GONG, revealed=True),
            (0, 1): Piece("b", Rank.QI, revealed=True),
            (11, 1): Piece("r", Rank.QI, revealed=True),
        }, turn=0, cfg=RuleConfig(flag_needs_mines_cleared=False, flag_gong_only=True))
        st.seat_color[0], st.seat_color[1] = "r", "b"
        st.first_flip_done = True

        scored = agent.choose_actions(st, topn=1)
        assert scored[0][0] == Action("move", (1, 1), (0, 1))
