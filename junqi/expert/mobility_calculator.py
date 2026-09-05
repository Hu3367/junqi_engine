"""
Expert-2: Space Layer - 空间与势力控制

实现棋理体系第 2 章的核心概念：
- Mobility-1/2/3 calculation (多层移动能力)
- Strategic node identification (战略节点识别)
- Space advantage scoring (空间优势评分)
- Graph theory concepts (图论工具)
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
import sys
sys.path.append('.')

from junqi.state import GameState, Piece
from junqi.config import RuleConfig

@dataclass
class MobilityReport:
    """移动能力报告"""
    mobility_1_count: int  # 1 步可达格数
    mobility_2_count: int  # 2 步可达格数
    mobility_3_count: int  # 3 步潜在可达格数
    rail_access: bool
    center_control: float  # 中心性评分 0-1
    
    @property
    def overall_mobility(self) -> float:
        """综合移动能力评分"""
        return (self.mobility_1_count * 0.5 + 
                self.mobility_2_count * 0.3 + 
                self.mobility_3_count * 0.2)


class MobilityCalculator:
    """
    移动能力计算器
    
    计算棋子在不同时间尺度下的可达区域
    """
    
    # 铁路网格定义
    RAIL_GRID = {
        (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),     # 上行铁路
        (9, 2), (9, 3), (9, 4), (9, 5), (9, 6),     # 下行铁路
        (2, 0), (3, 0), (4, 0),                      # 左列铁路
        (2, 8), (3, 8), (4, 8),                      # 右列铁路
        (5, 0), (5, 4), (5, 8),                      # 横向交汇点
        (6, 0), (6, 4), (6, 8)                       # 横向交汇点
    }
    
    CENTRAL_JUNCTIONS = [(5, 0), (5, 4), (6, 0), (6, 4)]
    
    def __init__(self, config: RuleConfig):
        self.config = config
    
    def calculate_mobility_1(self, board: GameState, piece_pos: Tuple[int, int],
                            piece_type: str) -> Set[Tuple[int, int]]:
        """
        计算 Mobility-1: 一回合内可达的所有格子
        
        Includes:
        - 公路移动（相邻四方向）
        - 铁路滑行（如果棋子可以在铁路上滑行）
        - 工兵飞行（如果适用）
        """
        reachable = set()
        
        dr_dc = [(-1,0), (1,0), (0,-1), (0,1)]
        
        for dr, dc in dr_dc:
            nr, nc = piece_pos[0] + dr, piece_pos[1] + dc
            
            # 检查是否在棋盘内
            if board.is_valid_position((nr, nc)):
                # 检查是否可以移动到这个位置
                if self._can_move_to(board, piece_pos, (nr, nc), piece_type):
                    reachable.add((nr, nc))
        
        # 如果是铁路，添加滑行路径
        if piece_type in self.config.RAIL_PIECES and piece_pos in self.RAIL_GRID:
            reachable.update(self._rail_slides_from(piece_pos, board))
        
        # 工兵的特殊移动（如果可以飞）
        if piece_type == '工兵' and self._can_fly(board, piece_pos):
            reachable.update(self._air_corridors_from(piece_pos, board))
        
        return reachable
    
    def calculate_mobility_2(self, board: GameState, piece_pos: Tuple[int, int],
                            piece_type: str) -> Set[Tuple[int, int]]:
        """
        计算 Mobility-2: 两回合内可达的区域
        
        关键点:
        1. 第一回合可能经过的行营/铁路
        2. 第二回合从这些中转点延伸
        3. 考虑敌方阻挡的可能性
        """
        m1 = self.calculate_mobility_1(board, piece_pos, piece_type)
        m2 = set()
        
        for intermediate_pos in m1:
            # 注意：intermediate_pos 处的棋子可能改变了，这里简化处理
            m1_from_intermediate = self.calculate_mobility_1(
                board, intermediate_pos, piece_type
            )
            m2.update(m1_from_intermediate)
        
        return m2
    
    def calculate_mobility_3(self, board: GameState, piece_pos: Tuple[int, int],
                            piece_type: str) -> Set[Tuple[int, int]]:
        """
        计算 Mobility-3: 三回合潜在可达区域
        
        考虑:
        1. 最乐观的移动路线（假设无阻挡）
        2. 基于信念的预测（某些暗子可能是障碍）
        3. 最佳路径规划（优先选择铁路/行营）
        """
        m2 = self.calculate_mobility_2(board, piece_pos, piece_type)
        m3 = set()
        
        for pos in m2:
            m1_from_pos = self.calculate_mobility_1(board, pos, piece_type)
            m3.update(m1_from_pos)
        
        return m3
    
    def _can_move_to(self, board: GameState, from_pos: Tuple[int, int],
                    to_pos: Tuple[int, int], piece_type: str) -> bool:
        """判断是否可以移动到目标位置"""
        # 检查目标是否有己方棋子
        target_piece = board.get_piece_at(to_pos)
        if target_piece is not None and target_piece.color == board.current_turn:
            return False
        
        # 检查是否是行营
        if board.board.is_bunker(to_pos):
            return True
        
        # 检查是否是大本营
        if board.board.is_base(to_pos):
            if piece_type not in ['司令', '军长']:
                return False
            if not board.board.is_my_base(to_pos, board.current_turn):
                return False
        
        return True
    
    def _rail_slides_from(self, start: Tuple[int, int], 
                         board: GameState) -> Set[Tuple[int, int]]:
        """获取从某个位置出发的所有铁路滑行可达点"""
        slides = set()
        
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # 水平垂直四个方向
        
        for dr, dc in directions:
            current = start
            while True:
                next_pos = (current[0] + dr, current[1] + dc)
                
                # 检查是否还在铁路线上
                if next_pos not in self.RAIL_GRID:
                    break
                
                # 不能越过边界
                if not board.is_valid_position(next_pos):
                    break
                
                # 检查是否有阻挡
                piece_at_next = board.get_piece_at(next_pos)
                if piece_at_next is not None:
                    # 可以吃到对方的棋子
                    slides.add(next_pos)
                    break
                else:
                    slides.add(next_pos)
                
                current = next_pos
        
        return slides
    
    def _can_fly(self, board: GameState, piece_pos: Tuple[int, int]) -> bool:
        """判断工兵是否可以飞行"""
        # 简单的飞行规则：如果前方有空白或敌棋
        row, col = piece_pos
        
        # 工兵可以从己方阵地起飞到远离的地带
        can_fly_positions = [
            (row - 1, col),  # 向前
            (row + 1, col),  # 向后（如果是在后方）
        ]
        
        return any(board.is_valid_position(p) for p in can_fly_positions)
    
    def _air_corridors_from(self, start: Tuple[int, int], 
                           board: GameState) -> Set[Tuple[int, int]]:
        """获取工兵可以飞行的目标位置"""
        corridors = set()
        
        # 简化版：只能飞到前线附近
        row, col = start
        
        for dc in range(-3, 4):
            target_col = col + dc
            if board.is_valid_position((row, target_col)):
                corridors.add((row, target_col))
        
        return corridors
    
    def get_mobility_report(self, board: GameState, piece_pos: Tuple[int, int],
                           piece_type: str) -> MobilityReport:
        """获取完整的移动能力报告"""
        m1 = self.calculate_mobility_1(board, piece_pos, piece_type)
        m2 = self.calculate_mobility_2(board, piece_pos, piece_type)
        m3 = self.calculate_mobility_3(board, piece_pos, piece_type)
        
        has_rail_access = piece_pos in self.RAIL_GRID or \
                         any(pos in self.RAIL_GRID for pos in m1)
        
        center_score = self._calculate_center_control(piece_pos)
        
        return MobilityReport(
            mobility_1_count=len(m1),
            mobility_2_count=len(m2),
            mobility_3_count=len(m3),
            rail_access=has_rail_access,
            center_control=center_score
        )


@dataclass
class StrategicNode:
    """战略节点"""
    name: str
    positions: List[Tuple[int, int]]
    base_value: float  # 0-1.0
    description: str

class StrategicNodeAnalyzer:
    """
    战略节点分析器
    
    识别棋盘上的关键位置并评估其实时价值
    """
    
    def __init__(self):
        self.nodes = self._initialize_nodes()
    
    def _initialize_nodes(self) -> List[StrategicNode]:
        """初始化所有已知的战略节点类型"""
        return [
            StrategicNode(
                name='central_rail_junctions',
                positions=[(5,0), (5,4), (6,0), (6,4)],
                base_value=0.95,
                description='中央铁路交汇点，连接南北快速通道'
            ),
            StrategicNode(
                name='flag_approach_paths',
                positions=[(1,0), (1,4), (10,0), (10,4)],
                base_value=0.90,
                description='军旗接近路径，工兵最后冲锋的关键路径'
            ),
            StrategicNode(
                name='frontline_crossings',
                positions=[(5,2), (6,2)],
                base_value=0.85,
                description='前线穿越点，唯一可直线穿越前线的公路'
            ),
            StrategicNode(
                name='camp_cluster_centers',
                positions=[(3,2), (8,2)],
                base_value=0.80,
                description='行营集群中心，前后方调度的枢纽'
            ),
            StrategicNode(
                name='railway_gates',
                positions=[(1,0), (1,4), (10,0), (10,4)],
                base_value=0.88,
                description='铁路闸门，控制一侧铁路线的入口'
            )
        ]
    
    def identify_controlled_nodes(self, board: GameState, color: int) -> List[Dict]:
        """
        识别当前被某方控制的战略节点
        
        Returns:
            List of [{'node': StrategicNode, 'control_level': float, ...}]
        """
        controlled = []
        
        for node in self.nodes:
            control_info = self._evaluate_node_control(node, board, color)
            controlled.append(control_info)
        
        # 按控制权排序
        controlled.sort(key=lambda x: x['control_level'], reverse=True)
        
        return controlled
    
    def _evaluate_node_control(self, node: StrategicNode, board: GameState,
                              color: int) -> Dict:
        """评估单个节点的控制程度"""
        controlled_count = 0
        contested_count = 0
        total_points = len(node.positions)
        
        for pos in node.positions:
            piece = board.get_piece_at(pos)
            
            if piece is None:
                # 空白 - 可能是潜在的
                continue
            elif piece.color == color:
                controlled_count += 1
            elif piece.is_revealed:
                # 对方明棋控制该点
                contested_count += 1
        
        control_level = controlled_count / max(total_points, 1)
        
        return {
            'node': node,
            'control_level': control_level,
            'controlled_points': controlled_count,
            'contested_points': contested_count,
            'total_points': total_points
        }
    
    def calculate_overall_space_control(self, board: GameState, 
                                       color: int) -> float:
        """
        计算某方在所有战略节点上的总体控制权
        
        Returns:
            float: -1.0 (完全劣势) to 1.0 (完全优势)
        """
        my_control = 0.0
        enemy_control = 0.0
        
        for node_info in self.identify_controlled_nodes(board, color):
            my_control += node_info['control_level']
            enemy_control += (1.0 - node_info['control_level'])
        
        avg_my_control = my_control / len(self.nodes)
        avg_enemy_control = enemy_control / len(self.nodes)
        
        # 归一化到 [-1, 1]
        diff = avg_my_control - avg_enemy_control
        
        return max(-1.0, min(1.0, diff))


class SpaceAdvantageCalculator:
    """
    空间优势计算器
    
    综合三个维度评估空间优劣：
    1. 活动空间 (Mobility sum)
    2. 控制密度 (Strategic nodes)
    3. 路径连通性 (Connectivity)
    """
    
    def __init__(self):
        self.node_analyzer = StrategicNodeAnalyzer()
        self.mobility_calc = MobilityCalculator(RuleConfig())
    
    def calculate_space_advantage(self, board: GameState, color: int) -> Dict:
        """
        计算双方在空间层面的优劣
        
        Returns:
            {
                'advantage_score': float,  # -1 to 1
                'my_mobility_ratio': float,
                'my_node_control': float,
                'my_connectivity': float
            }
        """
        # 1. 计算活动空间
        mobility_metrics = self._calculate_mobility_sum(board, color)
        mobility_ratio = mobility_metrics['ratio']
        
        # 2. 计算控制密度
        node_control = self.node_analyzer.calculate_overall_space_control(board, color)
        
        # 3. 计算路径连通性
        connectivity_ratio = self._calculate_connectivity(board, color)
        
        # 综合评分（加权平均）
        advantage_score = (
            normalize(mobility_ratio) * 0.4 +
            normalize(node_control) * 0.4 +
            normalize(connectivity_ratio) * 0.2
        )
        
        return {
            'advantage_score': advantage_score,
            'mobility_dimension': mobility_metrics,
            'control_dimension': node_control,
            'connectivity_dimension': connectivity_ratio,
            'overall_rating': self._interpret_score(advantage_score)
        }
    
    def _calculate_mobility_sum(self, board: GameState, color: int) -> Dict:
        """计算某方所有棋子的 Mobility-3 总和"""
        my_pieces = board.get_pieces(color)
        enemy_pieces = board.get_pieces(1 - color)
        
        my_total = 0.0
        enemy_total = 0.0
        
        for piece in my_pieces:
            if piece.is_revealed:
                report = self.mobility_calc.get_mobility_report(
                    board, piece.position, piece.type
                )
                my_total += report.overall_mobility
        
        for piece in enemy_pieces:
            if piece.is_revealed:
                report = self.mobility_calc.get_mobility_report(
                    board, piece.position, piece.type
                )
                enemy_total += report.overall_mobility
        
        ratio = my_total / max(enemy_total, 1.0)
        
        return {
            'my_sum': my_total,
            'enemy_sum': enemy_total,
            'ratio': ratio
        }
    
    def _calculate_connectivity(self, board: GameState, color: int) -> float:
        """
        计算某方棋子之间的连接效率
        
        Simplified version using graph theory
        """
        my_pieces = board.get_pieces(color)
        
        if len(my_pieces) <= 1:
            return 1.0  # 单一棋子默认完全连接
        
        # 构建连通图
        G = nx.Graph()
        
        for piece in my_pieces:
            G.add_node(piece.position)
            
            # 检查与其他棋子的连接
            for other_piece in my_pieces:
                if piece != other_piece:
                    dist = manhattan_distance(piece.position, other_piece.position)
                    if dist <= 3:  # 3 步范围内视为相连
                        G.add_edge(piece.position, other_piece.position)
        
        # 计算连通分量数量
        num_components = nx.number_connected_components(G)
        
        # 越多组件表示越分散，分数越低
        max_components = len(my_pieces)
        connectivity = 1.0 - (num_components / max_components)
        
        return max(0.0, connectivity)
    
    def _interpret_score(self, score: float) -> str:
        """解释空间优势评分"""
        if score >= 0.7:
            return "Overwhelming space advantage - Can attack aggressively"
        elif score >= 0.3:
            return "Moderate space advantage - Steady expansion recommended"
        elif score >= -0.2:
            return "Balanced position - Wait for opportunities"
        elif score >= -0.7:
            return "Space disadvantage - Consider defensive positioning"
        else:
            return "Critical space loss - Need immediate countermeasures"


# 工具函数

def normalize(value: float, min_val: float = 0.0, max_val: float = 3.0) -> float:
    """将值归一化到 [-1, 1]"""
    if max_val == min_val:
        return 0.0
    
    normalized = (value - min_val) / (max_val - min_val)
    return normalized * 2.0 - 1.0

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])