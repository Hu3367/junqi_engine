"""Tkinter 人机对战界面（参照手机 App 截图风格，棋子贴图取自 APK 原版资源）。

  python -m junqi gui

操作：点选己方明子 → 红框选中并高亮合法落点 → 点落点走子/吃子；
点暗子两次确认翻开。右侧可设 AI 棋力、发牌种子（同种子同布局，便于复现检验）、
查看剩余暗子构成和对局记录。提示按钮用 AI 给你推荐一步（金框标注）。
棋子贴图由 gen_assets.py 从 APK 精灵表切片生成（assets_app/cells/），
缺文件时自动回退到矢量绘制。
"""
from __future__ import annotations

import json
import math
import os
import queue
import random
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import messagebox

from .ai import Agent
from .config import EvalWeights, RuleConfig, SearchConfig
from .rules import (ALL_POSITIONS, CAMPS, COLOR_CN, HQS, NEIGHBORS, RANK_CN,
                    Rank, is_rail, rail_neighbors)
from .state import Action, GameState, Piece, deal, position_key

# ---------------------------------------------------------------- 配色（对照截图）
BG = "#BFE0F0"        # 天蓝背景
WOOD = "#D8AC6F"      # 木纹棋盘
WOOD_DARK = "#B98D52"
BACK_FILL = "#5E8C3C"   # 暗子背面绿
BACK_EDGE = "#3C6425"
R_FACE = "#DE812B"      # 红方(橙)棋面
B_FACE = "#4472A8"      # 蓝方棋面
FACE_TEXT = "white"
RAIL_FILL = "#222222"
ROAD_FILL = "#8B5E34"
SEL = "#EE1111"         # 选中红框（同截图）
TARGET = "#FF5050"
CONFIRM = "#FFD400"     # 翻子二次确认
HINT = "#FFC800"        # 提示金框

CELL_W, CELL_H = 88, 52
COLS, ROWS = 5, 12
MX, MY = 40, 26         # 棋盘边距
TITLE_H = 46
CENTER_GAP = 34         # 两军前线间隙

BOARD_W = MX * 2 + COLS * CELL_W
BOARD_H = TITLE_H + MY * 2 + ROWS * CELL_H + CENTER_GAP
PANEL_W = 300
WIN_W, WIN_H = BOARD_W + PANEL_W + 20, BOARD_H + 4

FONT = ("Microsoft YaHei", 11)
FONT_S = ("Microsoft YaHei", 9)
FONT_PIECE = ("Microsoft YaHei", 14, "bold")
FONT_TITLE = ("Microsoft YaHei", 22, "bold")

# APK 原版棋子贴图（gen_assets.py 生成；缺失时回退矢量绘制）
CELLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets_app", "cells")

REASON_CN = {
    "flag": "军旗被扛",
    "immobilized": "无棋可走",
    "no_capture": "连续70步未吃子，判和",
    "max_plies": "总步数用尽，判和",
    "repetition": "相同局面循环出现，判和",
    "draw_agreement": "双方协议和棋",
    "draw": "判定和棋",
    "resign": "认输",
}


def rc_to_xy(r, c):
    """格子中心像素。"""
    x = MX + c * CELL_W + CELL_W / 2
    y = TITLE_H + MY + r * CELL_H + CELL_H / 2 + (CENTER_GAP if r >= 6 else 0)
    return x, y


def xy_to_rc(x, y):
    for r in range(ROWS):
        for c in range(COLS):
            cx, cy = rc_to_xy(r, c)
            if abs(x - cx) <= CELL_W / 2 and abs(y - cy) <= CELL_H / 2:
                return (r, c)
    return None


def face_color(piece: Piece) -> str:
    return R_FACE if piece.color == "r" else B_FACE


def piece_label(piece: Piece) -> str:
    return f"{COLOR_CN[piece.color]}{RANK_CN[piece.rank]}"


class GuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("军棋翻棋 · 人机对战")
        root.configure(bg=BG)
        # 初始窗口尺寸与屏幕居中
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = max(0, (screen_w - WIN_W) // 2)
        y = max(0, (screen_h - WIN_H) // 2)
        root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
        root.resizable(False, False)
        self.cfg = RuleConfig()
        self.weights = EvalWeights()
        self.depth = tk.IntVar(value=2)
        self.samples = tk.IntVar(value=6)
        self.human_seat = 0
        self.busy = False
        self.pending_flip = None
        self.selected = None
        self.targets = []
        self.hint_action = None
        self.history = []
        self.log_lines = []
        self.result_q = queue.Queue()

        self.canvas = tk.Canvas(root, width=BOARD_W, height=BOARD_H,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(side="left")
        self.canvas.bind("<Button-1>", self.on_click)
        self._load_sprites()
        self._build_panel()
        self.new_game()

    def _load_sprites(self):
        """加载 APK 原版棋子贴图 {(color, Rank): PhotoImage} + 暗子背面。"""
        self.sprites = {}
        self.sprite_back = None
        try:
            files = sorted(os.listdir(CELLS_DIR))
        except OSError:
            return
        for f in files:
            if not f.endswith(".png") or f.startswith("_"):
                continue
            try:
                img = tk.PhotoImage(file=os.path.join(CELLS_DIR, f))
            except tk.TclError:
                continue
            stem = f[:-4]
            if stem == "back":
                self.sprite_back = img
            elif "_" in stem:
                color, rank = stem.split("_", 1)
                try:
                    self.sprites[(color, Rank[rank])] = img
                except KeyError:
                    continue

    # ------------------------------------------------------------- 右侧面板

    def _build_panel(self):
        p = tk.Frame(self.root, bg=BG, width=PANEL_W)
        p.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=8)

        tk.Label(p, text="军 棋 对 战", font=FONT_TITLE, bg=BG,
                 fg="#2F5D1E").pack(pady=(0, 4))
        self.status = tk.Label(p, text="", font=FONT, bg=BG, fg="#333",
                               wraplength=PANEL_W - 20, justify="left")
        self.status.pack(pady=2)

        row = tk.Frame(p, bg=BG)
        row.pack(pady=4)
        tk.Button(row, text="新局", font=FONT, width=5,
                  command=self.new_game).pack(side="left", padx=2)
        tk.Button(row, text="悔棋", font=FONT, width=5,
                  command=self.undo).pack(side="left", padx=2)
        tk.Button(row, text="提示", font=FONT, width=5,
                  command=self.ask_hint).pack(side="left", padx=2)
        tk.Button(row, text="求和", font=FONT, width=5,
                  command=self.offer_draw).pack(side="left", padx=2)

        row2 = tk.Frame(p, bg=BG)
        row2.pack(pady=2)
        tk.Button(row2, text="执先手", font=FONT_S,
                  command=lambda: self.new_game(human_seat=0)).pack(side="left", padx=3)
        tk.Button(row2, text="执后手", font=FONT_S,
                  command=lambda: self.new_game(human_seat=1)).pack(side="left", padx=3)

        self.ai_engine = tk.StringVar(value="p4_hybrid" if os.path.exists("models/best.pt") else "expert")
        adv_engine = tk.Frame(p, bg=BG)
        adv_engine.pack(pady=2)
        tk.Label(adv_engine, text="AI引擎:", font=FONT_S, bg=BG).pack(side="left")
        tk.Radiobutton(adv_engine, text="P4混合智能", variable=self.ai_engine, value="p4_hybrid",
                       font=FONT_S, bg=BG).pack(side="left", padx=1)
        tk.Radiobutton(adv_engine, text="专家搜索", variable=self.ai_engine, value="expert",
                       font=FONT_S, bg=BG).pack(side="left", padx=1)

        adv = tk.Frame(p, bg=BG)
        adv.pack(pady=2)
        tk.Label(adv, text="搜索算力:", font=FONT_S, bg=BG).pack(side="left")
        for text, d, k in (("快", 1, 4), ("标准", 2, 6), ("深算", 3, 4)):
            tk.Radiobutton(adv, text=text, variable=self.depth, value=d,
                            font=FONT_S, bg=BG,
                            command=lambda d=d, k=k: self.samples.set(k)
                            ).pack(side="left", padx=2)

        seed_row = tk.Frame(p, bg=BG)
        seed_row.pack(pady=2)
        tk.Label(seed_row, text="发牌种子:", font=FONT_S, bg=BG).pack(side="left")
        self.seed_var = tk.StringVar(value="")
        tk.Entry(seed_row, textvariable=self.seed_var, width=10,
                 font=FONT_S).pack(side="left", padx=4)
        tk.Label(seed_row, text="(空=随机)", font=FONT_S, bg=BG).pack(side="left")

        # 胜率预测组件
        wr_frame = tk.Frame(p, bg=BG)
        wr_frame.pack(fill="x", pady=(6, 2))
        self.winrate_label = tk.Label(wr_frame, text="胜率预测: 红方 50.0%  |  蓝方 50.0%",
                                      font=FONT_S, bg=BG, fg="#222")
        self.winrate_label.pack(anchor="w")
        self.winrate_bar = tk.Canvas(wr_frame, width=PANEL_W - 24, height=12,
                                     bg="#CCC", highlightthickness=1, highlightbackground="#999")
        self.winrate_bar.pack(pady=2)

        tk.Label(p, text="暗子剩余构成（公开信息推断）", font=FONT_S,
                 bg=BG, fg="#555").pack(anchor="w", pady=(6, 0))
        self.pool_text = tk.Text(p, height=6, width=36, font=FONT_S,
                                 bg="#EAF4FB", relief="flat")
        self.pool_text.pack(pady=2)

        tk.Label(p, text="对局记录", font=FONT_S, bg=BG, fg="#555").pack(anchor="w")
        log_frame = tk.Frame(p, bg=BG)
        log_frame.pack(fill="both", expand=True, pady=2)
        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side="right", fill="y")
        self.logbox = tk.Listbox(log_frame, font=FONT_S, bg="#F7F7F0",
                                 yscrollcommand=log_scroll.set)
        self.logbox.pack(side="left", fill="both", expand=True)
        log_scroll.config(command=self.logbox.yview)

    # ------------------------------------------------------------- 对局控制

    def new_game(self, human_seat: int | None = None):
        if human_seat is not None:
            self.human_seat = human_seat
        seed_text = self.seed_var.get().strip()
        seed = int(seed_text) if seed_text.isdigit() else random.randrange(10 ** 9)
        self.state = deal(random.Random(seed), self.cfg)
        self.history = []
        self.log_lines = []
        self.logbox.delete(0, "end")
        self.log("new", f"新局 种子={seed} 你执座位{self.human_seat}"
                        f"（{'先手' if self.human_seat == 0 else '后手'}）")
        # 对战记录（用于人类战法分析与审计回放，结束自动存 games/）
        self.record = {
            "seed": seed,
            "human_seat": self.human_seat,
            "engine_type": self.ai_engine.get(),
            "depth": self.depth.get(),
            "samples": self.samples.get(),
            "moves": [],
            "winner": None,
            "reason": None,
            "plies": 0,
        }
        self.busy = False
        self.pending_flip = None
        self.selected = None
        self.hint_action = None
        self.pos_seen = Counter([position_key(self.state)])   # 可观察局面计数（循环判和）
        self.gen = getattr(self, "gen", 0) + 1
        self.refresh()
        if self.state.turn != self.human_seat:
            self.root.after(400, self.ai_move)

    def _rebuild_seen(self):
        self.pos_seen = Counter(position_key(s) for s, _ in self.history)
        self.pos_seen[position_key(self.state)] += 1

    def _avoid_set(self) -> set:
        """再走一步就会触发循环判和的局面键集合（传给 AI 规避）。"""
        thr = self.cfg.repetition_draw_count - 1
        return {k for k, n in self.pos_seen.items() if n >= thr} if thr >= 1 else set()

    def undo(self):
        if self.busy or not self.history:
            return
        self.gen = getattr(self, "gen", 0) + 1
        while self.history:
            st, log_len = self.history.pop()
            self.state = st
            del self.log_lines[log_len:]
            if self.state.turn == self.human_seat:
                break
        self.logbox.delete(0, "end")
        for _, text in self.log_lines:
            self.logbox.insert("end", text)
        self.selected = self.pending_flip = self.hint_action = None
        self._rebuild_seen()
        self.refresh()

    def log(self, tag: str, text: str):
        self.log_lines.append((tag, text))
        self.logbox.insert("end", text)
        self.logbox.see("end")

    # ------------------------------------------------------------- 事件

    def on_click(self, event):
        if self.busy or self.state.is_terminal():
            return
        if self.state.turn != self.human_seat:
            return
        rc = xy_to_rc(event.x, event.y)
        if rc is None:
            return
        self.handle_cell(rc)

    def handle_cell(self, rc):
        st = self.state
        pc = st.board.get(rc)
        my = st.my_color()

        # 二次确认翻子
        if self.pending_flip is not None:
            if rc == self.pending_flip:
                self.do_action(Action("flip", rc))
            else:
                self.pending_flip = None
                self.selected = None
            self.refresh()
            return

        # 已选中己方明子：点落点执行
        if self.selected is not None:
            act = next((a for a in self.targets
                        if a.kind == "move" and a.to == rc), None)
            if act is not None:
                self.do_action(act)
                return
            if rc == self.selected:
                self.selected = None
                self.refresh()
                return

        # 点暗子：第一次选中待确认，第二次翻开
        if pc is not None and not pc.revealed:
            self.selected = None
            self.pending_flip = rc
            self.refresh()
            return

        # 点己方明子：选中并显示合法落点
        if pc is not None and pc.revealed and my and pc.color == my:
            acts = [a for a in st.legal_actions()
                    if a.kind == "move" and a.frm == rc]
            if acts:
                self.selected = rc
                self.targets = acts
                self.hint_action = None
                self.refresh()
            return
        self.selected = None
        self.refresh()

    def do_action(self, act: Action, ai_meta: dict | None = None):
        st = self.state
        desc = self.describe(act)
        self.history.append((st.copy(), len(self.log_lines)))
        move_rec = {
            "ply": st.ply, "seat": st.turn, "kind": act.kind,
            "frm": list(act.frm), "to": list(act.to) if act.to else None
        }
        if ai_meta:
            move_rec.update(ai_meta)
        self.record["moves"].append(move_rec)
        self.state = st.apply(act)
        self.pos_seen[position_key(self.state)] += 1
        self.selected = self.pending_flip = self.hint_action = None
        self.log("move", f"{'你' if st.turn == self.human_seat else 'AI'}: {desc}")
        self.refresh()
        if self.state.is_terminal():
            self.save_record()
            return
        if self.state.turn != self.human_seat:
            self.root.after(300, self.ai_move)

    def save_record(self):
        import hashlib
        model_p = "models/best.pt" if os.path.exists("models/best.pt") else "models/bc_best.pt"
        model_sha = None
        if os.path.exists(model_p):
            try:
                with open(model_p, "rb") as f:
                    model_sha = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                pass

        self.record.update(
            winner=self.state.winner,
            reason=self.state.win_reason,
            plies=self.state.ply,
            engine_type=self.ai_engine.get(),
            depth=self.depth.get(),
            samples=self.samples.get(),
            model_path=model_p if os.path.exists(model_p) else None,
            model_sha256=model_sha,
        )
        try:
            os.makedirs("games", exist_ok=True)
            name = time.strftime("games/game_%Y%m%d_%H%M%S") + ".json"
            with open(name, "w", encoding="utf-8") as f:
                json.dump(self.record, f, ensure_ascii=False, indent=1)
            self.log("save", f"对局记录已存 {name}")
        except OSError as e:
            self.log("save", f"记录保存失败: {e}")
    def describe(self, act: Action) -> str:
        st = self.state
        if act.kind == "flip":
            pc = st.board[act.frm]
            return f"翻({act.frm[0]},{act.frm[1]}) -> {piece_label(pc)}"
        mover = st.board[act.frm]
        target = st.board.get(act.to)
        base = f"({act.frm[0]},{act.frm[1]})->({act.to[0]},{act.to[1]})"
        if target is None:
            return f"{base} {RANK_CN[mover.rank]}"
        from .rules import battle
        res = battle(mover.rank, target.rank)
        if res == "attacker_wins":
            if target.rank == Rank.QI:
                return f"{base} 扛旗！吃掉{piece_label(target)}"
            return f"{base} {RANK_CN[mover.rank]}吃{piece_label(target)}"
        if res == "both_die":
            return f"{base} 同归于尽({RANK_CN[mover.rank]}×{piece_label(target)})"
        return f"{base} {RANK_CN[mover.rank]}撞{piece_label(target)}阵亡"

    # ------------------------------------------------------------- AI

    def ai_move(self):
        if self.busy or self.state.is_terminal():
            return
        self.busy = True
        self.gen = getattr(self, "gen", 0)
        self.status.config(text="AI 思考中…")
        st, seed = self.state.copy(), random.randrange(2 ** 30)
        depth, samples, gen = self.depth.get(), self.samples.get(), self.gen
        engine_type = self.ai_engine.get()
        avoid = self._avoid_set()

        def work():
            try:
                if engine_type == "p4_hybrid":
                    from .hybrid_engine import HybridDecisionEngine
                    mp = "models/best.pt" if os.path.exists("models/best.pt") else "models/bc_best.pt"
                    k = 4 if depth <= 2 else 8
                    agent = HybridDecisionEngine(model_path=mp, k_worlds=k, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid, history_counts=self.pos_seen)
                elif engine_type == "hybrid":
                    from .ai import HybridAgent
                    mp = "models/bc_best.pt" if os.path.exists("models/bc_best.pt") else "models/best.pt"
                    agent = HybridAgent(model_path=mp, search_depth=depth, weights=self.weights, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid)
                elif engine_type == "expert":
                    from .ai import ExpertAgent
                    time_budget = 400 if depth <= 1 else (1000 if depth == 2 else 2500)
                    agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_budget), weights=self.weights, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid)
                elif engine_type == "nn" and os.path.exists("models/best.pt"):
                    from .ai import NNAgent
                    sims = 15 if depth == 1 else (30 if depth == 2 else 50)
                    agent = NNAgent(model_path="models/best.pt", simulations=sims, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid)
                else:
                    agent = Agent(SearchConfig(depth=depth, samples=samples),
                                  weights=self.weights, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid)

                if not scored:
                    acts = st.legal_actions()
                    if acts:
                        scored = [(acts[0], 0.0)]

                best_act = scored[0][0] if scored else None
                ai_meta = {
                    "engine": engine_type,
                    "ai_seed": seed,
                    "top_scored": [(str(a), round(float(s), 2)) for a, s in scored[:3]],
                }
                self.result_q.put(("move", (best_act, ai_meta), gen))
            except Exception as e:
                import traceback
                traceback.print_exc()
                acts = st.legal_actions()
                emergency_act = acts[0] if acts else None
                ai_meta = {"engine": engine_type, "ai_seed": seed, "top_scored": [], "error": str(e)}
                self.result_q.put(("move", (emergency_act, ai_meta), gen))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(120, self._poll)

    def ask_hint(self):
        if self.busy or self.state.is_terminal() \
                or self.state.turn != self.human_seat:
            return
        self.busy = True
        self.gen = getattr(self, "gen", 0)
        self.status.config(text="计算提示中…")
        st, seed = self.state.copy(), random.randrange(2 ** 30)
        depth, samples, gen = self.depth.get(), self.samples.get(), self.gen
        engine_type = self.ai_engine.get()
        avoid = self._avoid_set()

        def work():
            try:
                if engine_type == "p4_hybrid":
                    from .hybrid_engine import HybridDecisionEngine
                    mp = "models/best.pt" if os.path.exists("models/best.pt") else "models/bc_best.pt"
                    k = 4 if depth <= 2 else 8
                    agent = HybridDecisionEngine(model_path=mp, k_worlds=k, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid, history_counts=self.pos_seen)
                elif engine_type == "hybrid":
                    from .ai import HybridAgent
                    mp = "models/bc_best.pt" if os.path.exists("models/bc_best.pt") else "models/best.pt"
                    agent = HybridAgent(model_path=mp, search_depth=depth, weights=self.weights, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid)
                elif engine_type == "expert":
                    from .ai import ExpertAgent
                    time_budget = 400 if depth <= 1 else (1000 if depth == 2 else 2500)
                    agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_budget), weights=self.weights, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid)
                elif engine_type == "nn" and os.path.exists("models/best.pt"):
                    from .ai import NNAgent
                    sims = 15 if depth == 1 else (30 if depth == 2 else 50)
                    agent = NNAgent(model_path="models/best.pt", simulations=sims, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid)
                else:
                    agent = Agent(SearchConfig(depth=depth, samples=samples),
                                  weights=self.weights, seed=seed)
                    scored = agent.choose_actions(st, topn=3, avoid=avoid)
                self.result_q.put(("hint", scored, gen))
            except Exception as e:
                import traceback
                traceback.print_exc()
                acts = st.legal_actions()
                self.result_q.put(("hint", [(acts[0], 0.0)] if acts else [], gen))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(120, self._poll)

    def _poll(self):
        try:
            kind, payload, gen = self.result_q.get_nowait()
        except queue.Empty:
            self.root.after(120, self._poll)
            return
        if gen != getattr(self, "gen", 0):      # 新局/悔棋后的过期结果，丢弃
            self.busy = False
            self.refresh()
            return
        self.busy = False
        if kind == "move" and payload is not None:
            act, ai_meta = payload
            if act is not None:
                self.do_action(act, ai_meta=ai_meta)
            else:
                self.refresh()
        elif kind == "hint":
            if payload:
                self.hint_action = payload[0][0]
                tips = "  |  ".join(f"{i}.{a} ({v:+.0f})"
                                    for i, (a, v) in enumerate(payload, 1))
                self.log("hint", f"提示: {tips}")
            self.refresh()

    # ------------------------------------------------------------- 绘制与胜率

    def refresh(self):
        self.draw_board()
        self.draw_pieces()
        self.draw_markers()
        self.update_status()
        self.update_pool()
        self.update_winrate()

    def get_winrate(self) -> tuple[float, float, float]:
        """计算当前局面 (红方胜率%, 和棋率%, 蓝方胜率%)。"""
        st = self.state
        if st.is_terminal():
            if st.winner == -1:
                return 0.0, 100.0, 0.0
            r_seat = 0 if st.seat_color.get(0) == "r" else (1 if st.seat_color.get(1) == "r" else None)
            if r_seat is not None:
                return (100.0, 0.0, 0.0) if st.winner == r_seat else (0.0, 0.0, 100.0)
            return (100.0, 0.0, 0.0) if st.winner == 0 else (0.0, 0.0, 100.0)

        # 0. 顶层结构性必和死锁断言（Dead Draw Assertion）
        from .analysis import is_dead_draw
        is_draw, _ = is_dead_draw(st)
        if is_draw:
            return 0.0, 100.0, 0.0

        engine_type = self.ai_engine.get() if hasattr(self, "ai_engine") else "expert"

        # 仅当显式选择 P4 混合智能时，才尝试调用神经网络模型
        if engine_type in ("p4_hybrid", "hybrid"):
            model_p = "models/best.pt" if os.path.exists("models/best.pt") else "models/bc_best.pt"
            if os.path.exists(model_p):
                try:
                    if not hasattr(self, "_cached_hybrid_engine") or getattr(self, "_cached_model_path", "") != model_p:
                        from .hybrid_engine import HybridDecisionEngine
                        self._cached_hybrid_engine = HybridDecisionEngine(model_path=model_p, k_worlds=4)
                        self._cached_model_path = model_p
                    info = self._cached_hybrid_engine.evaluate_position(st, history_counts=self.pos_seen)
                    p_win = info["win"]
                    p_draw = info["draw"]
                    p_loss = info["loss"]
                    turn_color = st.my_color()
                    if turn_color == "r":
                        p_red = p_win
                        p_blue = p_loss
                    elif turn_color == "b":
                        p_red = p_loss
                        p_blue = p_win
                    else:
                        rem = 1.0 - p_draw
                        p_red = 0.5 * rem
                        p_blue = 0.5 * rem
                    return p_red * 100.0, p_draw * 100.0, p_blue * 100.0
                except Exception:
                    pass

        # 专家搜索引擎估值映射（根据局面评分、残局时钟与和棋拓扑平滑映射）
        from .ai import evaluate_expert
        seat0_color = st.seat_color.get(0)
        score0 = evaluate_expert(st, seat=0, w=self.weights)
        if seat0_color == "r":
            red_score = score0
        elif seat0_color == "b":
            red_score = -score0
        else:
            red_score = 0.0

        # 根据残局阶段、无吃子步数动态估计和棋概率
        quiet = getattr(st, "quiet", 0)
        max_q = getattr(st.cfg, "no_capture_draw_plies", 70)
        progress = min(1.0, quiet / max_q) if max_q > 0 else 0.0
        p_draw_base = 0.20 + 0.60 * (progress ** 1.5)

        # 估值越接近 0，和棋概率越逼近 100%
        abs_score = abs(red_score)
        score_draw_factor = math.exp(-abs_score / 60.0)
        p_draw = min(0.99, p_draw_base + (1.0 - p_draw_base) * score_draw_factor)

        # 胜负概率分配
        p_red_raw = 1.0 / (1.0 + math.exp(-max(-600.0, min(600.0, red_score)) / 100.0))
        p_rem = max(0.01, 1.0 - p_draw)
        p_red = p_red_raw * p_rem
        p_blue = (1.0 - p_red_raw) * p_rem
        return p_red * 100.0, p_draw * 100.0, p_blue * 100.0

    def update_winrate(self):
        red_pct, draw_pct, blue_pct = self.get_winrate()
        self.winrate_label.config(text=f"胜率预测: 红胜 {red_pct:.1f}% | 和 {draw_pct:.1f}% | 蓝胜 {blue_pct:.1f}%")
        cv = self.winrate_bar
        cv.delete("all")
        w = PANEL_W - 24
        h = 12
        w_red = max(0, min(w, int(w * red_pct / 100.0)))
        w_draw = max(0, min(w - w_red, int(w * draw_pct / 100.0)))

        # 1. 红方胜率段
        if w_red > 0:
            cv.create_rectangle(0, 0, w_red, h, fill=R_FACE, outline="")
        # 2. 和棋概率段
        if w_draw > 0:
            cv.create_rectangle(w_red, 0, w_red + w_draw, h, fill="#9E9E9E", outline="")
        # 3. 蓝方胜率段
        if w_red + w_draw < w:
            cv.create_rectangle(w_red + w_draw, 0, w, h, fill=B_FACE, outline="")
        cv.create_line(w / 2, 0, w / 2, h, fill="#FFFFFF", width=2, dash=(2, 2))

    def offer_draw(self):
        if self.busy:
            return
        if self.state.is_terminal():
            messagebox.showinfo("提示", "当前对局已经结束！")
            return
        if self.state.turn != self.human_seat:
            messagebox.showinfo("提示", "请在你的行动回合提出求和！")
            return

        red_pct, draw_pct, blue_pct = self.get_winrate()
        ai_seat = 1 - self.human_seat
        ai_color = self.state.seat_color.get(ai_seat)
        if ai_color == "r":
            ai_win_pct = red_pct
        elif ai_color == "b":
            ai_win_pct = blue_pct
        else:
            ai_win_pct = 50.0

        # AI 智能求和判定：AI胜率 <= 55% 或 和棋率 >= 45% 时接受和棋
        if ai_win_pct <= 55.0 or draw_pct >= 45.0:
            self.state.winner = -1
            self.state.win_reason = "draw_agreement"
            self.log("draw", f"你提出了求和，AI 评估局势均衡（AI胜率 {ai_win_pct:.1f}%，和率 {draw_pct:.1f}%），同意和棋！")
            messagebox.showinfo("求和成功", f"AI 接受了你的求和请求！（AI预估胜率 {ai_win_pct:.1f}%，和率 {draw_pct:.1f}%）\n双方握手言和。")
            self.save_record()
            self.refresh()
        else:
            self.log("draw", f"你提出了求和，AI 认为自身占据优势（AI胜率 {ai_win_pct:.1f}%），拒绝了和棋。")
            messagebox.showinfo("求和被拒", f"AI 认为自身处于明显优势（胜率约 {ai_win_pct:.1f}%），拒绝和棋，请继续对局！")

    def draw_board(self):
        cv = self.canvas
        cv.delete("all")
        cv.create_rectangle(6, TITLE_H, BOARD_W - 6, BOARD_H - 6,
                            fill=WOOD, outline=WOOD_DARK, width=4)
        cv.create_text(BOARD_W / 2, TITLE_H / 2 + 4, text="军  棋",
                       font=FONT_TITLE, fill="#3A5F2A")

        # 公路（含行营斜道、前线三通道）
        for p, ns in NEIGHBORS.items():
            for q in ns:
                if p < q:
                    x0, y0 = rc_to_xy(*p)
                    x1, y1 = rc_to_xy(*q)
                    cv.create_line(x0, y0, x1, y1, fill=ROAD_FILL, width=3)
        # 铁路（黑白虚线感：黑粗虚线）
        for p in ALL_POSITIONS:
            if not is_rail(p):
                continue
            for q in rail_neighbors(p):
                if p < q:
                    x0, y0 = rc_to_xy(*p)
                    x1, y1 = rc_to_xy(*q)
                    cv.create_line(x0, y0, x1, y1, fill=RAIL_FILL,
                                   width=5, dash=(10, 6))

        for (r, c) in CAMPS:
            x, y = rc_to_xy(r, c)
            cv.create_oval(x - 24, y - 16, x + 24, y + 16, outline="#4C7A2E",
                           width=4)
        for (r, c) in HQS:                      # 大本营橙色括号（同截图）
            x, y = rc_to_xy(r, c)
            y_top = y - CELL_H / 2 + 2
            y_bot = y + CELL_H / 2 - 2
            cv.create_line(x - 26, y_top, x + 26, y_top, fill="#DE812B", width=3)
            cv.create_line(x - 26, y_bot, x + 26, y_bot, fill="#DE812B", width=3)

    def draw_pieces(self):
        cv = self.canvas
        use_img = bool(self.sprites)
        for pos, pc in self.state.board.items():
            x, y = rc_to_xy(*pos)
            img = (self.sprites.get((pc.color, pc.rank)) if pc.revealed
                   else self.sprite_back) if use_img else None
            if img is not None:
                cv.create_image(x, y, image=img)
            elif pc.revealed:
                x0, y0 = x - CELL_W / 2 + 4, y - CELL_H / 2 + 3
                x1, y1 = x + CELL_W / 2 - 4, y + CELL_H / 2 - 3
                cv.create_rectangle(x0, y0, x1, y1, fill=face_color(pc),
                                    outline="#222", width=1)
                cv.create_text(x, y, text=RANK_CN[pc.rank], font=FONT_PIECE,
                               fill=FACE_TEXT)
            else:
                x0, y0 = x - CELL_W / 2 + 4, y - CELL_H / 2 + 3
                x1, y1 = x + CELL_W / 2 - 4, y + CELL_H / 2 - 3
                cv.create_rectangle(x0, y0, x1, y1, fill=BACK_FILL,
                                    outline=BACK_EDGE, width=2)
                cv.create_rectangle(x0 + 5, y0 + 4, x1 - 5, y1 - 4,
                                    outline="#79A855", width=1)

    def draw_markers(self):
        cv = self.canvas
        if self.selected is not None:
            x, y = rc_to_xy(*self.selected)
            cv.create_rectangle(x - CELL_W / 2 + 2, y - CELL_H / 2 + 1,
                                x + CELL_W / 2 - 2, y + CELL_H / 2 - 1,
                                outline=SEL, width=3)
            for a in self.targets:
                x, y = rc_to_xy(*a.to)
                capture = self.state.board.get(a.to) is not None
                cv.create_rectangle(x - CELL_W / 2 + 2, y - CELL_H / 2 + 1,
                                    x + CELL_W / 2 - 2, y + CELL_H / 2 - 1,
                                    outline=TARGET, width=3 if capture else 2,
                                    dash=None if capture else (5, 3))
        if self.pending_flip is not None:
            x, y = rc_to_xy(*self.pending_flip)
            cv.create_rectangle(x - CELL_W / 2 + 2, y - CELL_H / 2 + 1,
                                x + CELL_W / 2 - 2, y + CELL_H / 2 - 1,
                                outline=CONFIRM, width=3)
            cv.create_text(x, y + CELL_H, text="再点确认翻开", font=FONT_S,
                           fill="#A06000")
        if self.hint_action is not None:
            for pos in [self.hint_action.frm] + (
                    [self.hint_action.to] if self.hint_action.to else []):
                x, y = rc_to_xy(*pos)
                cv.create_rectangle(x - CELL_W / 2 + 2, y - CELL_H / 2 + 1,
                                    x + CELL_W / 2 - 2, y + CELL_H / 2 - 1,
                                    outline=HINT, width=3)

    def update_status(self):
        st = self.state
        rule_line = (f"第 {st.ply}/{self.cfg.max_plies} 手 · "
                     f"无吃子 {st.quiet}/{self.cfg.no_capture_draw_plies}"
                     f" · 循环线 {self.cfg.repetition_draw_count} 次")
        if st.is_terminal():
            if st.winner == -1:
                outcome = f"和棋（{REASON_CN.get(st.win_reason, st.win_reason)}）"
            else:
                who = "你赢了！" if st.winner == self.human_seat else "AI 获胜"
                outcome = f"{who}（{REASON_CN.get(st.win_reason, st.win_reason)}）"
            text = f"{outcome} — 点【新局】再来\n{rule_line}"
        elif st.turn == self.human_seat:
            my = st.my_color()
            who = f"你执{COLOR_CN[my]}" if my else "你（首翻定色）"
            text = f"轮到你（{who}）：点己方明子走子，或点暗子翻开\n{rule_line}"
        else:
            text = f"轮到 AI…\n{rule_line}"
        self.status.config(text=text)

    def update_pool(self):
        rem = self.state.remaining_types()
        self.pool_text.config(state="normal")
        self.pool_text.delete("1.0", "end")
        for color in ("r", "b"):
            items = [f"{RANK_CN[r]}×{rem.get((color, r), 0)}"
                     for r in Rank if rem.get((color, r))]
            self.pool_text.insert("end",
                                  f"{COLOR_CN[color]}方暗子: " + " ".join(items) + "\n")
        self.pool_text.config(state="disabled")


def main():
    root = tk.Tk()
    GuiApp(root)
    root.mainloop()


launch_gui = main


if __name__ == "__main__":
    main()
