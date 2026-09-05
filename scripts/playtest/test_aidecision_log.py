"""
Junqi AI Decision Logger
用于验证规则文档并记录 AI 决策过程
"""

import random
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import deque


class Rank(IntEnum):
    """棋子等级体系 - 第 1.2 节"""
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
    
    def __str__(self):
        if not self.revealed:
            return "???"
        name_map = {
            Rank.GONG: "GONG", Rank.PAI: "PAI", Rank.LIAN: "LIAN",
            Rank.YING: "YING", Rank.TUAN: "TUAN", Rank.LV: "LV",
            Rank.SHI: "SHI", Rank.JUN: "JUN", Rank.SI: "SI",
            Rank.ZHA: "ZHA", Rank.LEI: "LEI", Rank.QI: "QI"
        }
        return f"{name_map[self.rank]}"


class Board:
    """棋盘 - 第 1.1 节"""
    
    def __init__(self):
        self.rows = 12
        self.cols = 5
        self.board: Dict[Tuple[int, int], Piece] = {}
        
        # 特殊区域
        self.camps = frozenset({
            (2, 1), (2, 3), (3, 2), (4, 1), (4, 3),  # 上方
            (7, 1), (7, 3), (8, 2), (9, 1), (9, 3),  # 下方
        })
        self.hqs = frozenset({(0, 1), (0, 3), (11, 1), (11, 3)})
        self.rail_rows = frozenset({1, 5, 6, 10})
        self.rail_cols = frozenset({0, 4})
        self.cross_blocked = frozenset({
            frozenset(((5, 1), (6, 1))),
            frozenset(((5, 2), (6, 2))),
            frozenset(((5, 3), (6, 3))),
        })
        
    def deal(self, seed: int = None) -> List[Tuple[int, int]]:
        """发牌 - 返回所有位置"""
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
        
        # 分配红黑棋子
        hidden_positions = []
        for i, pos in enumerate(positions[:25]):
            piece = Piece('r', red_ranks[i])
            self.board[pos] = piece
            hidden_positions.append(pos)
        for i, pos in enumerate(positions[25:]):
            piece = Piece('b', black_ranks[i])
            self.board[pos] = piece
            hidden_positions.append(pos)
            
        print(f"🎲 发牌完成 (种子={seed}, 暗子数={len(hidden_positions)})")
        return hidden_positions
        
    def display(self):
        """显示棋盘"""
        print("\n" + "="*50)
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
                else:
                    separator = "-" if (r, c) in self.rail_rows else "|" 
                    
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
                    else:
                        char = " "
                
                line += f" {char} │"
            print(line)
            
        print("="*50)
        
    @staticmethod
    def rank_to_char(rank: Rank) -> str:
        """等级转字符"""
        chars = {
            Rank.GONG: "工", Rank.PAI: "排", Rank.LIAN: "连",
            Rank.YING: "营", Rank.TUAN: "团", Rank.LV: "旅",
            Rank.SHI: "师", Rank.JUN: "军", Rank.SI: "司",
            Rank.ZHA: "炸", Rank.LEI: "雷", Rank.QI: "旗"
        }
        return chars.get(rank, "?")


class GameState:
    """游戏状态 - 第 1.5 节"""
    
    def __init__(self, board: Board, turn: str = 'r'):
        self.board = board
        self.turn = turn  # 当前行动方
        self.flipped_first = False  # 首次翻子标志
        self.first_color = None  # 首次翻子的颜色
        self.plies = 0  # 总步数
        self.last_capture_ply = 0  # 最后吃子步数
        self.hidden_count = 25 * 2  # 初始暗子数
        self.dead_pieces = set()  # 阵亡棋子构成
        
    def get_hidden_positions(self) -> List[Tuple[int, int]]:
        """获取所有暗子位置"""
        return [pos for pos, piece in self.board.board.items() if not piece.revealed]
        
    def can_flip(self, pos: Tuple[int, int]) -> bool:
        """是否可以翻某个暗子"""
        if pos not in self.board.board:
            return False
        piece = self.board.board[pos]
        return not piece.revealed and piece.color == self.turn
        
    def get_legal_moves(self) -> List[str]:
        """获取合法走法（简化版）"""
        moves = []
        
        # 1. 翻棋（只要有暗子）
        for pos, piece in self.board.board.items():
            if not piece.revealed and piece.color == self.turn:
                moves.append(f"flip:{pos}")
        
        # 2. 移动明子（简化版）
        for pos, piece in self.board.board.items():
            if piece.revealed and piece.color == self.turn:
                # 简化的邻居检查
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = pos[0]+dr, pos[1]+dc
                    if 0 <= nr < 12 and 0 <= nc < 5:
                        neighbor = self.board.board.get((nr, nc))
                        if neighbor is None:
                            moves.append(f"move:{pos}->{(nr,nc)}")
                        elif neighbor.color != piece.color and neighbor.revealed:
                            # 可以吃对方
                            res = self.battle(piece.rank, neighbor.rank)
                            if res in ["大吃小", "扛旗成功", "工兵挖雷成功"]:
                                moves.append(f"attack:{pos}->{(nr,nc)}")
                                
        return moves
        
    def apply_action(self, action: str):
        """执行动作"""
        parts = action.split(':')
        op = parts[0]
        
        if op == "flip":
            pos = eval(parts[1])
            self.flip_piece(pos)
        elif op == "move" or op == "attack":
            from_pos, to_pos = eval(parts[1]), eval(parts[2])
            self.move_piece(from_pos, to_pos)
            
    def flip_piece(self, pos: Tuple[int, int]):
        """翻棋子"""
        piece = self.board.board[pos]
        
        if not self.flipped_first:
            # 首次翻子决定阵营
            piece.revealed = True
            self.flipped_first = True
            self.first_color = self.turn
            piece.color = self.turn
            print(f"✅ [首次翻子] {piece.color}方执{('红','黑')[piece.color=='b']}")
        else:
            # 只能翻己方暗子
            piece.revealed = True
            print(f"✅ [翻子] 位置{pos}: {piece}")
            
        self.hidden_count -= 1
        
    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """移动棋子（简化版）"""
        piece = self.board.board[from_pos]
        target = self.board.board.get(to_pos)
        
        if target:
            if target.color != piece.color and target.revealed:
                # 战斗结算
                result = self.battle(piece.rank, target.rank)
                print(f"⚔️ [战斗] {piece} {result} {target}")
                self.board.board.pop(to_pos)
                
                if target.rank != Rank.QI:  # 军旗不进入阵亡池
                    self.dead_pieces.add((target.color, target.rank))
                    if target.rank == Rank.ZHA:
                        self.board.board.pop(from_pos)
                        
            else:
                print(f"❌ [移动] 目标位置非法")
                return
        else:
            # 移动到空位
            pass
            
        self.board.board[to_pos] = piece
        del self.board.board[from_pos]
        
    @staticmethod
    def battle(attacker: Rank, defender: Rank) -> str:
        """战斗结算 - 第 1.3 节"""
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
        
    def check_game_over(self) -> Optional[str]:
        """检查游戏结束"""
        # flag captured
        for pos, piece in self.board.board.items():
            if piece.revealed and piece.rank == Rank.QI:
                if piece.color != self.turn:
                    winner = "红" if piece.color == 'r' else "黑"
                    return f"🏁 {winner}方扛旗获胜!"
                    
        # max plies
        if self.plies >= 1000:
            return "🏁 总步数 1000，和棋!"
            
        return None


class SimpleAI:
    """简单 AI - 基于基础规则策略"""
    
    def __init__(self, game: GameState, color: str):
        self.game = game
        self.color = color
        self.decision_log = []
        
    def make_decision(self) -> str:
        """做出决策"""
        log = {
            "turn": self.game.turn,
            "action_type": None,
            "rationale": "",
            "hidden_count": self.game.hidden_count,
            "flipped_first": self.game.flipped_first
        }
        
        # 阶段判定
        phase = self._detect_phase()
        log["phase"] = phase
        
        # 策略选择
        if phase == "opening":
            action = self._opening_strategy()
        elif phase == "midgame":
            action = self._midgame_strategy()
        else:
            action = self._endgame_strategy()
            
        log["selected_move"] = action
        log["rationale"] = f"[{phase}] {log['rationale']}"
        
        self.decision_log.append(log)
        print(f"🤖 [AI决策] {log['rationale']}")
        print(f"💡 [推荐] 执行：{action}")
        
        return action
        
    def _detect_phase(self) -> str:
        """阶段判定 - 文档第 2-4 节"""
        h = self.game.hidden_count
        
        if h >= 20:
            return "opening"
        elif h >= 6:
            return "midgame"
        else:
            return "endgame"
            
    def _opening_strategy(self) -> str:
        """开局策略 - 文档第 2 节"""
        hidden = self.game.get_hidden_positions()
        
        # 策略 1: 优先翻前线
        front_positions = [(r, c) for r, c in hidden if 5 <= r <= 6]
        
        # 策略 2: 选择安全性高的
        safe_positions = []
        for pos in front_positions:
            safety_score = self._calculate_flip_safety(pos)
            safe_positions.append((pos, safety_score))
            
        if safe_positions:
            best = max(safe_positions, key=lambda x: x[1])
            log_msg = f"翻前线安全位{best[0]} (安全评分:{best[1]:.0f})"
            return f"flip:{best[0]}"
        else:
            # fallback
            pos = random.choice(hidden[:5])
            log_msg = f"随机翻边路{pos}"
            return f"flip:{pos}"
            
    def _calculate_flip_safety(self, pos: Tuple[int, int]) -> float:
        """计算翻棋安全评分"""
        score = 0.0
        
        # 周围友军保护
        guards = 0
        enemy_threats = 0
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0]+dr, pos[1]+dc
            if 0 <= nr < 12 and 0 <= nc < 5:
                piece = self.game.board.board.get((nr, nc))
                if piece and piece.revealed:
                    if piece.color == self.color:
                        guards += 1
                    else:
                        enemy_threats += 1
                        
        if guards > enemy_threats:
            score = 60 + (guards - enemy_threats)
        elif enemy_threats > guards:
            score = 1 - enemy_threats
        else:
            score = 20
            
        return score
        
    def _midgame_strategy(self) -> str:
        """中盘策略 - 文档第 3 节"""
        moves = self.game.get_legal_moves()
        
        if not moves:
            return "flip:" + str(self.game.get_hidden_positions()[0])
            
        # 优先选择吃子
        attacks = [m for m in moves if m.startswith("attack")]
        if attacks:
            selected = random.choice(attacks)
            return selected
            
        # 其次翻棋
        flips = [m for m in moves if m.startswith("flip")]
        if flips:
            return random.choice(flips)
            
        # 否则移动
        return random.choice(moves)
        
    def _endgame_strategy(self) -> str:
        """尾盘策略 - 文档第 4 节"""
        moves = self.game.get_legal_moves()
        
        if not moves:
            return "flip:" + str(self.game.get_hidden_positions()[0])
            
        # 寻找最佳路径
        attacks = [m for m in moves if m.startswith("attack")]
        if attacks:
            return random.choice(attacks)
            
        return random.choice(moves)


def play_game():
    """主游戏循环"""
    print("="*60)
    print("🎮 军棋翻棋对弈测试 - AI 决策过程记录")
    print("="*60)
    
    # 初始化
    board = Board()
    seed = input("\n请输入发牌种子（直接回车随机）: ").strip()
    seed = int(seed) if seed else random.randint(1, 9999)
    
    print(f"\n使用种子：{seed}")
    
    hidden = board.deal(seed)
    
    game = GameState(board, 'r')
    ai_red = SimpleAI(game, 'r')
    ai_black = SimpleAI(game, 'b')
    
    player_color = input("\n你想执哪一方？(r/b): ").strip().lower()
    if player_color not in ['r', 'b']:
        player_color = 'r'
        
    print(f"\n{'='*60}")
    print(f"玩家执{('红','黑')[player_color=='b']}方，AI 执{('黑','红')[player_color=='b']}方")
    print(f"{'='*60}")
    
    # 主循环
    while True:
        current_turn = game.turn
        is_player_turn = current_turn == player_color
        
        # 显示棋盘
        game.board.display()
        
        # 显示信息
        phase = ai_red._detect_phase()
        print(f"\n📊 [阶段] {phase} | 暗子数：{game.hidden_count} | 回合：{game.plies}")
        print(f"💬 [AI 决策日志即将记录到 aid_decision_log.json]")
        
        # 获取行动
        if is_player_turn:
            # 玩家行动
            action = input("\n你的行动 (flip/r/c): ").strip()
            
            if action.lower() == 'c':
                print("退出游戏")
                break
                
            elif action.lower() == 'f':
                pos_str = input("输入坐标 (如 5,2): ").strip()
                try:
                    pos = eval(pos_str)
                    action = f"flip:{pos}"
                except:
                    print("❌ 无效坐标，重试")
                    continue
                    
            elif action.lower() == 'r':
                from_pos = input("输入起点 (如 5,0): ").strip()
                to_pos = input("输入终点 (如 4,0): ").strip()
                try:
                    action = f"move:{eval(from_pos)}:{eval(to_pos)}"
                except:
                    print("❌ 无效坐标，重试")
                    continue
                    
        else:
            # AI 行动
            ai = ai_red if current_turn == 'r' else ai_black
            action = ai.make_decision()
            
        # 执行动作
        try:
            game.apply_action(action)
            game.plies += 1
            
            # 检查游戏结束
            result = game.check_game_over()
            if result:
                print(f"\n{result}")
                break
                
        except Exception as e:
            print(f"❌ 错误：{e}")
            print("请重试")
            continue
        
        # 切换回合
        game.turn = 'b' if current_turn == 'r' else 'r'
        
        # 保存 AI 决策日志
        with open("ai_decision_log.json", "w", encoding="utf-8") as f:
            import json
            # 收集所有 AI 的决策日志
            all_logs = []
            all_logs.extend(ai_red.decision_log)
            all_logs.extend(ai_black.decision_log)
            json.dump(all_logs, f, ensure_ascii=False, indent=2)
            
    print(f"\n📄 AI 决策日志已保存到：ai_decision_log.json")


if __name__ == "__main__":
    play_game()
