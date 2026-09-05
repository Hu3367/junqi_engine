"""
Junqi Game Engine for Playtesting
用于验证规则文档的策略合理性
"""

import random
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


class Rank(IntEnum):
    """棋子等级体系"""
    GONG = 5   # 工兵
    PAI = 6    # 排长
    LIAN = 7   # 连长
    YING = 8   # 营长
    TUAN = 9   # 团长
    LV = 10    # 旅长
    SHI = 11   # 师长
    JUN = 12   # 军长
    SI = 13    # 司令
    ZHA = 20   # 炸弹
    LEI = 21   # 地雷
    QI = 22    # 军旗


@dataclass
class Piece:
    """棋子"""
    color: str  # 'r' (红) or 'b' (黑)
    rank: Rank
    revealed: bool = False
    camp_pos: Optional[Tuple[int, int]] = None  # 是否在行营内


class Board:
    """简化版棋盘 - 仅用于快速测试"""
    
    def __init__(self):
        self.rows = 12
        self.cols = 5
        self.board: Dict[Tuple[int, int], Piece] = {}
        
        # 特殊区域
        self.camps = {
            (2,1), (2,3), (3,2), (4,1), (4,3),  # 上方
            (7,1), (7,3), (8,2), (9,1), (9,3),  # 下方
        }
        self.hqs = {(0,1), (0,3), (11,1), (11,3)}
        self.rail_rows = {1, 5, 6, 10}
        self.rail_cols = {0, 4}
        
    def deal(self, seed: int = None):
        """发牌"""
        if seed is not None:
            random.seed(seed)
            
        # 构建双方棋子
        composition = [
            ('SI', 1), ('JUN', 1), ('SHI', 2), ('LV', 2), ('TUAN', 2),
            ('YING', 2), ('LIAN', 3), ('PAI', 3), ('GONG', 3),
            ('ZHA', 2), ('LEI', 3), ('QI', 1)
        ]
        
        ranks = []
        for name, count in composition:
            for _ in range(count):
                ranks.append(getattr(Rank, name))
        
        red_ranks = ranks.copy()
        black_ranks = ranks.copy()
        
        random.shuffle(red_ranks)
        random.shuffle(black_ranks)
        
        # 填充棋盘（跳过行营）
        positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.camps:
                    positions.append((r, c))
        
        # 分配红黑棋子（前 5 行红，后 5 行黑，中间作为缓冲）
        for i, pos in enumerate(positions[:25]):
            self.board[pos] = Piece('r', red_ranks[i])
        for i, pos in enumerate(positions[25:]):
            self.board[pos] = Piece('b', black_ranks[i])
            
        print(f"🎲 发牌完成 (种子={seed})")
        
    def display(self):
        """显示棋盘"""
        print("\n" + "="*40)
        print("     0    1    2    3    4")
        
        for r in range(self.rows):
            line = f"{r:2d} │"
            for c in range(self.cols):
                piece = self.board.get((r, c))
                
                # 判断格子类型
                if (r, c) in self.camps:
                    char = "(●)"  # 行营
                elif (r, c) in self.hqs:
                    char = "[]"   # 大本营
                elif (r, c) in self.rail_rows or (r, c) in self.rail_cols:
                    separator = "-" if (r, c) in self.rail_rows else "|"
                else:
                    separator = "."
                
                # 显示棋子
                if piece:
                    if piece.revealed:
                        rank_name = self.rank_to_char(piece.rank)
                        prefix = "R" if piece.color == 'r' else "B"
                        char = f"{prefix}{rank_name}"
                    else:
                        char = "??"  # 暗子
                else:
                    if (r, c) in self.camps:
                        char = "( )"
                    elif (r, c) in self.hqs:
                        char = "[ ]"
                    elif (r, c) in self.rail_rows or (r, c) in self.rail_cols:
                        char = separator
                    else:
                        char = " "
                
                line += f" {char} │"
            print(line)
            
        print("="*40)
        
    @staticmethod
    def rank_to_char(rank: Rank) -> str:
        """等级转字符"""
        chars = {
            Rank.GONG: "G", Rank.PAI: "P", Rank.LIAN: "L",
            Rank.YING: "Y", Rank.TUAN: "T", Rank.LV: "V",
            Rank.SHI: "S", Rank.JUN: "J", Rank.SI: "K",
            Rank.ZHA: "X", Rank.LEI: "M", Rank.QI: "F"
        }
        return chars.get(rank, "?")


class Game:
    """简化版对弈引擎 - 用于文档验证"""
    
    def __init__(self):
        self.board = Board()
        self.turn = 'r'  # 红先
        self.flipped_first = False
        self.plies = 0
        
    def start(self):
        """开始游戏"""
        seed = input("请输入发牌种子（直接回车随机）: ").strip()
        seed = int(seed) if seed else random.randint(1, 9999)
        
        self.board.deal(seed)
        self.board.display()
        
    def play_turn(self):
        """执行一回合"""
        player_color = self.turn
        
        while True:
            action = input(f"\n{player_color.upper()}方行动 (flip/r/c 翻子/移动/退出): ")
            
            if action.lower() == 'c':
                print("退出游戏")
                break
                
            elif action.lower() == 'f':
                # 翻子逻辑（简化）
                pos = self.parse_position(input("输入坐标 (如 5,2): "))
                self.flip_piece(pos)
                
            elif action.lower() == 'r':
                # 移动逻辑（简化）
                from_pos = self.parse_position(input("输入起始位置 (如 5,0): "))
                to_pos = self.parse_position(input("输入目标位置 (如 4,0): "))
                self.move_piece(from_pos, to_pos)
                
            self.plies += 1
            self.board.display()
            
            if self.check_game_over():
                break
                
        # 切换回合
        self.turn = 'b' if self.turn == 'r' else 'r'
        
    def flip_piece(self, pos: Tuple[int, int]):
        """翻开棋子"""
        if pos not in self.board.board:
            print("❌ 无效位置")
            return
            
        piece = self.board.board[pos]
        
        # 第一次翻子决定阵营
        if not self.flipped_first:
            piece.revealed = True
            self.flipped_first = True
            piece.color = 'r' if self.turn == 'r' else 'b'
            print(f"✅ 首次翻子! {piece.color}方执红/黑")
        else:
            piece.revealed = True
            print(f"✅ 翻开{pos}: {self.board.rank_to_char(piece.rank)}")
            
    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """移动棋子（简化版）"""
        if from_pos not in self.board.board:
            print("❌ 起点无效")
            return
            
        piece = self.board.board[from_pos]
        
        if not piece.revealed:
            print("❌ 只能移动明子")
            return
            
        if piece.color != self.turn:
            print(f"❌ 只能移动{self.turn}方的棋子")
            return
            
        # 简单移动（不检查合法性）
        target = self.board.board.get(to_pos)
        
        if target:
            if target.color != piece.color and target.revealed:
                # 战斗
                result = self.battle(piece.rank, target.rank)
                print(f"⚔️ 战斗结果:{result}")
                del self.board.board[to_pos]
            else:
                print("❌ 不能移动到自己棋子位置")
                return
        else:
            # 移动到空位
            pass
            
        # 移动棋子
        self.board.board[to_pos] = piece
        del self.board.board[from_pos]
        
    @staticmethod
    def battle(attacker: Rank, defender: Rank) -> str:
        """战斗结算"""
        if defender == Rank.QI:
            return "扛旗成功!"
        if attacker == Rank.ZHA or defender == Rank.ZHA:
            return "同归于尽!"
        if defender == Rank.LEI:
            if attacker == Rank.GONG:
                return "工兵挖雷成功!"
            else:
                return "攻击地雷失败!"
        if attacker == defender:
            return "同级同尽!"
        return "大吃小" if attacker > defender else "被吃"
        
    def parse_position(self, s: str) -> Tuple[int, int]:
        """解析坐标"""
        try:
            r, c = map(int, s.strip().split(','))
            return (r, c)
        except:
            return None
            
    def check_game_over(self):
        """检查游戏结束（简化）"""
        if self.plies >= 1000:
            print("🏁 总步数 1000，和棋!")
            return True
        return False


# ============ 启动对弈 ============
if __name__ == "__main__":
    game = Game()
    game.start()
    
    while True:
        game.play_turn()
        if len(game.board.board) == 0:
            break
