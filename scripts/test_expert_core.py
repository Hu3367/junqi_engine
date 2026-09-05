#!/usr/bin/env python3
"""
Expert Engine V1.0 快速验证脚本

这个脚本用来测试核心组件的基本功能是否正常工作。
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from junqi.config import RuleConfig
from junqi.expert import ExpertEngine, MonteCarloTreeSearch

def test_imports():
    """测试所有模块能否正常导入"""
    print("\n" + "="*60)
    print("Test: Import Verification")
    print("="*60)
    
    try:
        from junqi.expert import (
            RuleValidator,
            TacticalAnalyzer,
            ThreatDetectionEngine,
            ConditionalPieceValueEvaluator,
            BeliefSystem,
            MobilityCalculator,
            SpaceAdvantageCalculator,
            TempoTracker,
            PieceValueReport,
            Threat,
            ThreatTier,
            MonteCarloTreeSearch,
            ExpertEngine
        )
        print("✓ All expert modules imported successfully!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_rule_validator_basic():
    """测试规则验证器基础功能"""
    print("\n" + "="*60)
    print("Test: Rule Validator Basic")
    print("="*60)
    
    config = RuleConfig()
    
    from junqi.expert.rule_validator import RuleValidator
    validator = RuleValidator(config)
    
    # 创建基础配置验证
    if config.PIECE_RANKS:
        print(f"✓ RuleConfig loaded with {len(config.PIECE_RANKS)} piece types")
        print(f"  Example: Marshal={config.PIECE_RANKS.get('司令', 'N/A')}")
        return True
    else:
        print("✗ RuleConfig empty or invalid")
        return False

def test_tactical_analyzer_structure():
    """测试战术分析器结构"""
    print("\n" + "="*60)
    print("Test: Tactical Analyzer Structure")
    print("="*60)
    
    from junqi.expert.rule_validator import RuleValidator
    from junqi.expert.tactical_analyzer import TacticalAnalyzer, ThreatAssessment
    
    config = RuleConfig()
    validator = RuleValidator(config)
    analyzer = TacticalAnalyzer(validator)
    
    # 检查主要方法存在
    methods = ['analyze_position', 'detect_capture_opportunities', 
              'detect_enemy_threats', 'identify_forced_moves']
    
    for method in methods:
        if hasattr(analyzer, method):
            print(f"✓ Method {method} exists")
        else:
            print(f"✗ Method {method} missing")
            return False
    
    print("✓ TacticalAnalyzer structure verified!")
    return True

def test_benefit_system_structure():
    """测试信念系统结构"""
    print("\n" + "="*60)
    print("Test: Belief System Structure")
    print("="*60)
    
    from junqi.expert.hidden_piece_belief import HiddenPieceBelief, BeliefSystem
    
    # 检查类和方法
    if hasattr(HiddenPieceBelief, 'calculate_entropy'):
        print("✓ HiddenPieceBelief.calculate_entropy exists")
    else:
        print("✗ calculate_entropy missing")
        return False
    
    if hasattr(BeliefSystem, 'get_highest_uncertainty_positions'):
        print("✓ BeliefSystem.get_highest_uncertainty_positions exists")
    else:
        print("✗ get_highest_uncertainty_positions missing")
        return False
    
    print("✓ Belief System structure verified!")
    return True

def test_mobility_calculator_structure():
    """测试移动能力计算器结构"""
    print("\n" + "="*60)
    print("Test: Mobility Calculator Structure")
    print("="*60)
    
    from junqi.expert.mobility_calculator import MobilityCalculator
    
    config = RuleConfig()
    calc = MobilityCalculator(config)
    
    # 检查核心属性
    required_attrs = ['RAIL_GRID', 'CENTRAL_JUNCTIONS']
    for attr in required_attrs:
        if hasattr(calc, attr):
            value = getattr(calc, attr)
            print(f"✓ {attr}: {type(value).__name__} (length={len(value) if hasattr(value, '__len__') else 'N/A'})")
        else:
            print(f"✗ {attr} missing")
            return False
    
    # 检查核心方法
    methods = ['calculate_mobility_1', 'calculate_mobility_2', 'calculate_mobility_3']
    for method in methods:
        if hasattr(calc, method):
            print(f"✓ Method {method} exists")
        else:
            print(f"✗ Method {method} missing")
            return False
    
    print("✓ Mobility Calculator structure verified!")
    return True

def test_tempo_tracker_structure():
    """测试 Tempo 追踪器结构"""
    print("\n" + "="*60)
    print("Test: Tempo Tracker Structure")
    print("="*60)
    
    from junqi.expert.tempo_tracker import TempoTracker, TempoFlowType
    
    tracker = TempoTracker()
    
    # 检查枚举值
    flow_types = ['GAIN', 'LOSS', 'FORCED', 'RESERVE']
    for ft in flow_types:
        if hasattr(TempoFlowType, ft):
            print(f"✓ TempoFlowType.{ft} exists")
        else:
            print(f"✗ TempoFlowType.{ft} missing")
            return False
    
    # 检查主要方法
    methods = ['record_move', 'get_current_assessment', 'calculate_tempo_with_threats']
    for method in methods:
        if hasattr(tracker, method):
            print(f"✓ Method {method} exists")
        else:
            print(f"✗ Method {method} missing")
            return False
    
    print("✓ Tempo Tracker structure verified!")
    return True

def test_threat_detection_structure():
    """测试威胁检测系统结构"""
    print("\n" + "="*60)
    print("Test: Threat Detection Structure")
    print("="*60)
    
    from junqi.expert.threat_detection import ThreatDetectionEngine, ThreatTier
    
    # 检查威胁等级
    tiers = ['T1_IMMEDIATE_CAPTURE', 'T2_CHAIN_ATTACK', 'T3_FLAG_ZONE',
             'T4_RAIL_INVASION', 'T5_ENCIRCLEMENT', 'T6_INFO_TRAP', 'T7_DEFERRED']
    
    for tier in tiers:
        if hasattr(ThreatTier, tier):
            print(f"✓ ThreatTier.{tier} exists")
        else:
            print(f"✗ ThreatTier.{tier} missing")
            return False
    
    # 检查引擎类
    if hasattr(ThreatDetectionEngine, 'detect_all_threats'):
        print("✓ ThreatDetectionEngine.detect_all_threats exists")
    else:
        print("✗ detect_all_threats missing")
        return False
    
    print("✓ Threat Detection system structure verified!")
    return True

def test_conditional_value_structure():
    """测试条件子力价值评估器结构"""
    print("\n" + "="*60)
    print("Test: Conditional Value Evaluator Structure")
    print("="*60)
    
    from junqi.expert.conditional_value import ConditionalPieceValueEvaluator, PieceValueReport
    
    config = RuleConfig()
    evaluator = ConditionalPieceValueEvaluator(config)
    
    # 检查五维度方法
    dimensions = [
        '_evaluate_position_quality',
        '_evaluate_offensive_pressure',
        '_evaluate_defensive_necessity',
        '_evaluate_future_mobility',
        '_evaluate_special_capabilities'
    ]
    
    for dim in dimensions:
        if hasattr(evaluator, dim):
            print(f"✓ Dimension method {dim} exists")
        else:
            print(f"✗ Dimension method {dim} missing")
            return False
    
    # 检查动态价值计算
    if hasattr(PieceValueReport, 'dynamic_value'):
        print("✓ PieceValueReport.dynamic_value property exists")
    else:
        print("✗ dynamic_value property missing")
        return False
    
    print("✓ Conditional Value Evaluator structure verified!")
    return True

def test_search_optimizer_structure():
    """测试搜索优化器结构"""
    print("\n" + "="*60)
    print("Test: Search Optimizer Structure")
    print("="*60)
    
    config = RuleConfig()
    mcts = MonteCarloTreeSearch(config, max_depth=4)
    
    # 检查主要方法
    methods = ['search', 'initialize_with_principles', '_select', '_expand', 
               '_simulate', '_backpropagate']
    
    for method in methods:
        if hasattr(mcts, method):
            print(f"✓ MCTS method {method} exists")
        else:
            print(f"✗ MCTS method {method} missing")
            return False
    
    print("✓ Search Optimizer structure verified!")
    return True

def test_expert_engine_integration():
    """测试专家引擎集成"""
    print("\n" + "="*60)
    print("Test: Expert Engine Integration")
    print("="*60)
    
    try:
        config = RuleConfig()
        engine = ExpertEngine(config)
        
        # 检查各组件已正确初始化
        components = ['rule_validator', 'tactical_analyzer', 
                     'space_calculator', 'tempo_tracker', 
                     'value_evaluator']
        
        for comp in components:
            if hasattr(engine, comp):
                print(f"✓ Component '{comp}' initialized")
            else:
                print(f"✗ Component '{comp}' not found")
                return False
        
        # 检查评估方法
        if hasattr(engine, 'evaluate_position'):
            print("✓ evaluate_position method exists")
        else:
            print("✗ evaluate_position method missing")
            return False
        
        if hasattr(engine, 'generate_training_sample'):
            print("✓ generate_training_sample method exists")
        else:
            print("✗ generate_training_sample method missing")
            return False
        
        print("✓ Expert Engine integration verified!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("EXPERT ENGINE V1.0 - QUICK VALIDATION SUITE")
    print("="*60)
    
    results = []
    
    tests = [
        ("Import Verification", test_imports),
        ("Rule Validator Basic", test_rule_validator_basic),
        ("Tactical Analyzer Structure", test_tactical_analyzer_structure),
        ("Belief System Structure", test_benefit_system_structure),
        ("Mobility Calculator Structure", test_mobility_calculator_structure),
        ("Tempo Tracker Structure", test_tempo_tracker_structure),
        ("Threat Detection Structure", test_threat_detection_structure),
        ("Conditional Value Structure", test_conditional_value_structure),
        ("Search Optimizer Structure", test_search_optimizer_structure),
        ("Expert Engine Integration", test_expert_engine_integration),
    ]
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n✗ {name} FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # 总结
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else f"✗ FAIL"
        print(f"{status:10s} {name}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Expert Engine V1.0 structure is valid.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
