#!/usr/bin/env python3
"""
Mini-Junqi 实验引擎

用于大规模自动对弈和统计分析，生成训练数据和最优策略洞察
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from junqi.mini_junqi import (
    MiniGameState, MiniMove, MoveType, deal_mini_pieces, MiniBoard
)
from junqi.mini_agents import BaseAgent, create_agent


@dataclass
class GameResult:
    """单局比赛结果"""
    winner: int  # 0 or 1
    win_reason: str
    total_turns: int
    red_final_material: int
    black_final_material: int
    move_history: List[Dict]
    
    # 统计数据
    flip_stats: Dict = field(default_factory=dict)
    capture_stats: Dict = field(default_factory=dict)
    bunker_control_stats: Dict = field(default_factory=dict)


class MiniExperimentEngine:
    """
    Mini-Junqi 实验引擎
    
    负责：
    1. 大规模自动对弈
    2. 数据收集与统计
    3. 策略分析
    """
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.board = MiniBoard(self.config)
        self.results = []
        self.stats_db = defaultdict(lambda: {'games': 0, 'wins': 0})
        
    def run_game(self, agent_red: BaseAgent, agent_black: BaseAgent,
                scheme_name: str = "scheme_a", max_turns: int = 60) -> GameResult:
        """
        运行一局比赛
        
        Args:
            agent_red: 红方 Agent
            agent_black: 黑方 Agent  
            scheme_name: 子力配置方案
            max_turns: 最大回合数
            
        Returns:
            GameResult: 比赛结果
        """
        # 初始化游戏状态
        state = self._initialize_game(scheme_name)
        
        move_history = []
        turn = 0
        
        while turn < max_turns and not state.winner:
            current_player = state.current_player
            moves = state.generate_legal_moves(current_player)
            
            if not moves:
                break
            
            # 选择移动
            if current_player == 0:
                agent = agent_red
            else:
                agent = agent_black
            
            selected_move = agent.select_move(state, moves)
            if not selected_move:
                break
            
            # 记录移动
            move_record = {
                'turn': turn,
                'player': current_player,
                'move': str(selected_move),
                'from_pos': selected_move.from_pos,
                'to_pos': selected_move.to_pos,
                'move_type': selected_move.move_type.value
            }
            
            # 应用移动
            new_state = state.apply(selected_move)
            move_history.append(move_record)
            
            # 检查胜负
            if new_state.winner:
                state = new_state
                break
            
            state = new_state
            turn += 1
        
        # 构建结果
        result = self._build_result(state, move_history, turn)
        self.results.append(result)
        
        return result
    
    def _initialize_game(self, scheme_name: str) -> MiniGameState:
        """初始化游戏"""
        # 发牌
        red_pieces, black_pieces = deal_mini_pieces(scheme_name, self.config)
        
        # 合并所有棋子
        all_pieces = red_pieces + black_pieces
        
        # 填充棋盘
        board_dict = {}
        for piece in all_pieces:
            board_dict[piece.position] = piece
        
        # 创建状态
        state = MiniGameState(
            board=board_dict,
            pieces=all_pieces,
            turn=0,
            current_player=0,
            first_flip_done=False
        )
        
        return state
    
    def _build_result(self, state: MiniGameState, 
                     move_history: List[Dict],
                     total_turns: int) -> GameResult:
        """构建比赛结果"""
        final_material = {
            0: sum(p.rank_value for p in state.get_pieces(0) if p.revealed),
            1: sum(p.rank_value for p in state.get_pieces(1) if p.revealed)
        }
        
        return GameResult(
            winner=state.winner,
            win_reason=state.win_reason,
            total_turns=total_turns,
            red_final_material=final_material[0],
            black_final_material=final_material[1],
            move_history=move_history
        )
    
    def run_batch(self, num_games: int, agent_type_red: str, 
                 agent_type_black: str, scheme_name: str = "scheme_a",
                 parallel: bool = False) -> List[GameResult]:
        """批量运行比赛"""
        print(f"\nRunning {num_games} games...")
        print(f"Red Agent: {agent_type_red}")
        print(f"Black Agent: {agent_type_black}")
        print(f"Scheme: {scheme_name}")
        print("="*60)
        
        results = []
        
        for i in range(num_games):
            # 创建新 Agent 实例
            agent_red = create_agent(agent_type_red, 0)
            agent_black = create_agent(agent_type_black, 1)
            
            # 设置随机种子
            random.seed(i * 42)
            
            # 运行游戏
            result = self.run_game(agent_red, agent_black, scheme_name)
            results.append(result)
            
            # 进度显示
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{num_games} games")
        
        print(f"\n✓ Batch complete: {len(results)} games played")
        
        return results
    
    def analyze_results(self, results: List[GameResult]) -> Dict:
        """分析结果并生成统计报告"""
        if not results:
            return {}
        
        analysis = {
            'total_games': len(results),
            'win_rates': {'red': 0.0, 'black': 0.0, 'draw': 0.0},
            'avg_turns': 0.0,
            'win_methods': defaultdict(int),
            'flip_positions': defaultdict(int),
            'bunker_occupancy': defaultdict(int),
            'material_advantage_analysis': {}
        }
        
        # 聚合统计
        total_turns = 0
        red_wins = 0
        black_wins = 0
        
        for result in results:
            # 胜负统计
            if result.winner == 0:
                red_wins += 1
            elif result.winner == 1:
                black_wins += 1
            
            total_turns += result.total_turns
            
            # 获胜方式统计
            if result.win_reason:
                analysis['win_methods'][result.win_reason] += 1
            
            # 翻棋位置统计
            for move in result.move_history:
                if move['move_type'] == 'flip':
                    pos_key = str(move['from_pos'])
                    analysis['flip_positions'][pos_key] += 1
            
            # 行营控制统计
            # TODO: 详细实现
        
        # 计算平均值
        analysis['avg_turns'] = total_turns / len(results)
        analysis['win_rates']['red'] = red_wins / len(results)
        analysis['win_rates']['black'] = black_wins / len(results)
        
        # 转换为列表便于 JSON 序列化
        analysis['win_methods'] = dict(analysis['win_methods'])
        analysis['flip_positions'] = dict(sorted(
            analysis['flip_positions'].items(),
            key=lambda x: x[1], reverse=True
        )[:10])  # Top 10 位置
        
        # 材料优势分析
        material_wins = [
            r for r in results 
            if r.win_reason == 'material_advantage' or 
               r.win_reason == 'material_disadvantage'
        ]
        
        if material_wins:
            analysis['material_advantage_analysis'] = {
                'count': len(material_wins),
                'percentage': len(material_wins) / len(results)
            }
        
        return analysis
    
    def save_report(self, analysis: Dict, output_path: str):
        """保存分析报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_games': analysis['total_games'],
                'avg_turns': round(analysis['avg_turns'], 2),
                'win_rates': analysis['win_rates']
            },
            'detailed': analysis
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Report saved to: {output_path}")
    
    def generate_strategy_insights(self, analysis: Dict) -> str:
        """生成策略洞察报告"""
        insights = []
        
        # 翻棋策略洞察
        if analysis['flip_positions']:
            top_flips = list(analysis['flip_positions'].items())[:5]
            insights.append("\n【翻棋位置最优解】")
            insights.append("Top 5 高胜率翻棋位置:")
            for pos, count in top_flips:
                insights.append(f"  - {pos}: {count}次 ({count/analysis['total_games']*100:.1f}%)")
        
        # 获胜方式洞察
        if analysis['win_methods']:
            insights.append("\n【获胜方式分布】")
            for method, count in sorted(
                analysis['win_methods'].items(), 
                key=lambda x: x[1], reverse=True
            ):
                insights.append(f"  - {method}: {count}次 ({count/analysis['total_games']*100:.1f}%)")
        
        # 平均回合数洞察
        avg_turns = analysis['avg_turns']
        if avg_turns < 20:
            insights.append("\n【游戏节奏】快速对局 (平均{}回合)".format(avg_turns))
        elif avg_turns < 40:
            insights.append("\n【游戏节奏】标准节奏 (平均{}回合)".format(avg_turns))
        else:
            insights.append("\n【游戏节奏】持久战 (平均{}回合)".format(avg_turns))
        
        return "\n".join(insights)


def main():
    parser = argparse.ArgumentParser(description="Mini-Junqi Experiment Engine")
    parser.add_argument('--games', type=int, default=1000, help='Number of games to play')
    parser.add_argument('--red-agent', type=str, default='heuristic', help='Red agent type')
    parser.add_argument('--black-agent', type=str, default='random', help='Black agent type')
    parser.add_argument('--scheme', type=str, default='scheme_a', help='Piece configuration scheme')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    
    args = parser.parse_args()
    
    # 使用默认配置路径
    config_path = "configs/mini_junqi_config.json"
    
    engine = MiniExperimentEngine(config_path)
    
    # 运行批量实验
    results = engine.run_batch(
        num_games=args.games,
        agent_type_red=args.red_agent,
        agent_type_black=args.black_agent,
        scheme_name=args.scheme
    )
    
    # 分析结果
    analysis = engine.analyze_results(results)
    
    # 保存报告
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"reports/mini_junqi_{timestamp}.json"
    
    engine.save_report(analysis, args.output)
    
    # 打印策略洞察
    print("\n" + "="*60)
    print(engine.generate_strategy_insights(analysis))
    print("="*60)


if __name__ == "__main__":
    main()
