"""局面计算器：命令行交互，录入/编辑局面、走子、悔棋、TopN 走法建议、自动推演。

棋子代码：颜色字母 + 等级字
  颜色：r=红 b=蓝
  等级：司 军 师 旅 团 营 连 排 工 炸 雷 旗
  ?? = 未翻开的暗子
示例：set 5,2 b师   /   open 3,1 r工   /   best 5
"""
from __future__ import annotations

import random
import sys

from .ai import Agent, win_probability
from .config import EvalWeights, RuleConfig, SearchConfig
from .rules import CAMPS, COLORS, RANK_CN, Rank, battle
from .state import Action, GameState, Piece, deal

RANK_CHAR = {Rank.SI: "司", Rank.JUN: "军", Rank.SHI: "师", Rank.LV: "旅",
             Rank.TUAN: "团", Rank.YING: "营", Rank.LIAN: "连", Rank.PAI: "排",
             Rank.GONG: "工", Rank.ZHA: "炸", Rank.LEI: "雷", Rank.QI: "旗"}
CHAR_RANK = {v: k for k, v in RANK_CHAR.items()}

HELP = """命令：
  deal [种子]              新开真实发牌局
  load 文件 / save 文件    JSON 读 / 写局面
  set 行,列 代码           放子（代码如 r师 / b雷 / ?? ）
  rm 行,列                 移除棋子
  open 行,列 代码          翻子事件（把暗子翻开为指定棋子）
  move 行1,列1 行2,列2     走子（自动战斗结算）
  turn 0|1                 设定行动方座位
  color r|b                设定当前行动方执的颜色（另一座位自动取反色）
  undo                     悔棋（可多次）
  best [N] [深度] [采样K]  前 N 个走法建议（默认 3 2 6）
  play [手数]              双方用搜索自动走 N 手（默认 1）
  show                     显示棋盘
  help / quit
"""


class Calculator:
    def __init__(self):
        self.cfg = RuleConfig()
        self.search = SearchConfig()
        self.weights = EvalWeights()
        self.state: GameState = deal(random.Random(), self.cfg)
        self.history: list[GameState] = []
        self.rng = random.Random()

    # ------------------------------------------------------------- 工具

    def push(self):
        self.history.append(self.state.copy())

    def parse_pos(self, token):
        parts = token.replace("，", ",").split(",")
        if len(parts) != 2:
            raise ValueError(f"坐标格式应为 行,列：{token}")
        r, c = int(parts[0]), int(parts[1])
        if not (0 <= r < 12 and 0 <= c < 5):
            raise ValueError(f"坐标越界：({r},{c})")
        return (r, c)

    def parse_piece(self, code) -> Piece:
        code = code.strip()
        if code == "??":
            return Piece(self.rng.choice(COLORS), Rank.GONG, False)  # 占位，身份由公开推断
        if len(code) != 2 or code[0] not in COLORS or code[1] not in CHAR_RANK:
            raise ValueError(f"棋子代码格式：r师/b雷/??，收到 {code}")
        return Piece(code[0], CHAR_RANK[code[1]], True)

    # ------------------------------------------------------------- 命令

    def cmd(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True
        parts = line.split()
        op, args = parts[0].lower(), parts[1:]
        try:
            if op in ("quit", "q", "exit"):
                return False
            elif op == "help":
                print(HELP)
            elif op == "show":
                print(self.state.render())
            elif op == "deal":
                seed = int(args[0]) if args else self.rng.randrange(10 ** 9)
                self.push()
                self.state = deal(random.Random(seed), self.cfg)
                print(f"新局，种子 {seed}")
                print(self.state.render())
            elif op == "save" and args:
                with open(args[0], "w", encoding="utf-8") as f:
                    f.write(self.state.to_json())
                print(f"已保存 {args[0]}")
            elif op == "load" and args:
                with open(args[0], encoding="utf-8") as f:
                    self.push()
                    self.state = GameState.from_json(f.read(), self.cfg)
                print(self.state.render())
            elif op == "set" and len(args) >= 2:
                pos = self.parse_pos(args[0])
                pc = self.parse_piece(args[1])
                self.push()
                b = dict(self.state.board)
                b[pos] = pc
                self.state = GameState(b, self.state.dead,
                                       dict(self.state.seat_color), self.state.turn,
                                       self.state.first_flip_done, self.state.ply,
                                       None, None, self.cfg)
                print(f"放置 {args[1]} @ {pos}")
            elif op == "rm" and args:
                pos = self.parse_pos(args[0])
                self.push()
                b = dict(self.state.board)
                b.pop(pos, None)
                self.state = GameState(b, self.state.dead,
                                       dict(self.state.seat_color), self.state.turn,
                                       self.state.first_flip_done, self.state.ply,
                                       None, None, self.cfg)
                print(f"移除 {pos}")
            elif op == "open" and len(args) >= 2:
                pos = self.parse_pos(args[0])
                pc = self.parse_piece(args[1])
                if pos not in self.state.board:
                    raise ValueError(f"{pos} 没有棋子")
                self.push()
                from dataclasses import replace
                b = dict(self.state.board)
                b[pos] = replace(pc, revealed=True)
                sc = dict(self.state.seat_color)
                ffd = self.state.first_flip_done
                if not ffd:
                    sc[self.state.turn] = pc.color
                    sc[1 - self.state.turn] = "b" if pc.color == "r" else "r"
                    ffd = True
                self.state = GameState(b, self.state.dead, sc,
                                       1 - self.state.turn, ffd,
                                       self.state.ply + 1, None, None, self.cfg)
                print(f"翻子 {pos} -> {RANK_CN[pc.rank]}({pc.color})")
            elif op == "move" and len(args) >= 2:
                frm = self.parse_pos(args[0])
                to = self.parse_pos(args[1])
                act = Action("move", frm, to)
                if act not in self.state.legal_actions():
                    raise ValueError("非法走法（可先 best 查看合法建议）")
                self.push()
                self.state = self.state.apply(act)
                print(self.state.render())
            elif op == "turn" and args:
                t = int(args[0])
                if t not in (0, 1):
                    raise ValueError("座位是 0 或 1")
                self.push()
                s = self.state.copy()
                s.turn = t
                self.state = s
            elif op == "color" and args:
                col = args[0].lower()
                if col not in COLORS:
                    raise ValueError("颜色是 r 或 b")
                self.push()
                s = self.state.copy()
                s.seat_color = {self.state.turn: col,
                                1 - self.state.turn: "b" if col == "r" else "r"}
                s.first_flip_done = True
                self.state = s
            elif op == "undo":
                if not self.history:
                    print("没有可悔的棋")
                else:
                    self.state = self.history.pop()
                    print(self.state.render())
            elif op == "best":
                topn = int(args[0]) if args else 3
                if len(args) >= 2:
                    self.search.depth = int(args[1])
                if len(args) >= 3:
                    self.search.samples = int(args[2])
                agent = Agent(self.search, self.weights, seed=self.rng.randrange(2 ** 30))
                scored = agent.choose_actions(self.state, topn=topn)
                if not scored:
                    print("无合法走法（可能已终局）")
                seat = self.state.turn
                for i, (a, v) in enumerate(scored, 1):
                    note = ""
                    if a.kind == "move":
                        t = self.state.board.get(a.to)
                        if t is not None and t.revealed:
                            res = {"attacker_wins": "可吃", "both_die": "同尽",
                                   "defender_wins": "必输"}[
                                battle(self.state.board[a.frm].rank, t.rank)]
                            note = f" 攻击{t.name}[{res}]"
                        elif t is None:
                            note = " 移动"
                    else:
                        note = " 翻开暗子"
                    print(f"{i}. {a}{note}  评分 {v:+.1f}  胜率~{win_probability(v):.0%}")
            elif op == "play":
                n = int(args[0]) if args else 1
                agent = Agent(self.search, self.weights, seed=self.rng.randrange(2 ** 30))
                for _ in range(n):
                    if self.state.is_terminal():
                        print("对局已结束")
                        break
                    self.push()
                    scored = agent.choose_actions(self.state, topn=1)
                    if not scored:
                        break
                    a = scored[0][0]
                    print(f"座位{self.state.turn}: {a}")
                    self.state = self.state.apply(a)
                print(self.state.render())
            else:
                print("未知命令，输入 help 查看帮助")
        except (ValueError, KeyError, IndexError) as e:
            print(f"错误：{e}")
        return True

    def loop(self):
        print("军棋翻棋局面计算器（输入 help 查看命令）")
        print(self.state.render())
        while True:
            try:
                line = input("junqi> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not self.cmd(line):
                break


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    Calculator().loop()
