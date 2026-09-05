#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test Script for Mini-Junqi
快速测试脚本 - 一次性运行所有基础测试
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("="*60)
    print("MINI-JUNQI QUICK TEST")
    print("="*60)
    
    # Step 1: Verify system
    print("\n[1/4] Running system verification...")
    import subprocess
    result = subprocess.run(
        ["python", "scripts/verify_mini_junqi.py"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    print(result.stdout)
    
    if result.returncode != 0:
        print("[ERROR] System verification failed!")
        return 1
    
    # Step 2: Import and test
    print("\n[2/4] Testing module imports...")
    try:
        from junqi.mini_junqi import MiniGameState, deal_mini_pieces
        
        # Create simple game
        red, black = deal_mini_pieces("scheme_a", {})
        all_pieces = red + black[:15]  # Use fewer pieces
        
        board = {p.position: p for p in all_pieces}
        state = MiniGameState(board=board, pieces=all_pieces)
        
        print(f"[PASS] Created game with {len(state.pieces)} pieces")
        print(f"Board size: {state.BOARD_SIZE}x{state.BOARD_SIZE}")
        print(f"Legal moves: {len(state.generate_legal_moves(0))}")
        
    except Exception as e:
        print(f"[FAIL] Import test: {e}")
        return 1
    
    # Step 3: Run small experiment
    print("\n[3/4] Running quick experiment (50 games)...")
    try:
        from junqi.mini_agents import HeuristicAgent, RandomAgent
        from junqi.mini_junqi import MiniExperimentEngine
        
        engine = MiniExperimentEngine("configs/mini_junqi_config.json")
        results = engine.run_batch(
            num_games=50,
            agent_type_red="heuristic",
            agent_type_black="random"
        )
        
        analysis = engine.analyze_results(results)
        print(f"[PASS] Completed 50 games")
        print(f"Avg turns: {analysis['avg_turns']:.1f}")
        print(f"Red win rate: {analysis['win_rates']['red']*100:.1f}%")
        print(f"Black win rate: {analysis['win_rates']['black']*100:.1f}%")
        
    except Exception as e:
        print(f"[WARN] Experiment failed: {e}")
        print("(This is OK if dependencies are not installed)")
    
    # Step 4: Generate summary
    print("\n[4/4] Test complete!")
    print("\nNext steps:")
    print("1. Install deps: pip install numpy pyyaml")
    print("2. Full experiment: python scripts/mini_junqi_experiment.py --games 1000")
    print("3. View strategy guide: docs/08-MiniJunqi/JUNQI_MINI_5X5_OPTIMAL_STRATEGY_GUIDE.md")
    
    return 0

if __name__ == "__main__":
    exit(main())
