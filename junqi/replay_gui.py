"""可视化军棋对局复盘与决策点评工作台 (Replay & Review GUI)。

使用方法：
    python -m junqi replay_gui
    python -m junqi replay_gui --file games/game_20260906_230423.json
    python -m junqi replay_gui --file "军旗复盘/251122223849 棋手 62240-棋手 44374.sav"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from .config import EvalWeights, RuleConfig, SearchConfig
from .gui import (ALL_POSITIONS, BACK_EDGE, BACK_FILL, BG, BOARD_H, BOARD_W,
                  B_FACE, CAMPS, CELLS_DIR, CELL_H, CELL_W, COLOR_CN, COLS,
                  FACE_TEXT, FONT, FONT_PIECE, FONT_S, FONT_TITLE, HINT, HQS,
                  MX, MY, NEIGHBORS, RAIL_FILL, RANK_CN, ROAD_FILL, ROWS,
                  R_FACE, SEL, TARGET, TITLE_H, WOOD, WOOD_DARK, face_color,
                  is_rail, piece_label, rail_neighbors, rc_to_xy, xy_to_rc)
from .replay_manager import ReplayManager, ReplaySession, ReplayStep
from .review_storage import REVIEW_LABELS, ReviewStorage
from .review_summary import ReviewSummaryPipeline
from .rules import Rank, other
from .state import Action, GameState, Piece

PANEL_W = 440
WIN_W = BOARD_W + PANEL_W + 28
WIN_H = BOARD_H + 52

PLAYED_COLOR = "#E60000"
REC_COLOR = "#27AE60"
BLUNDER_COLOR = "#E63946"


class ReplayViewerApp:
    def __init__(self, root: tk.Tk, initial_file: Optional[str] = None):
        self.root = root
        self.root.title("军棋对局复盘与算法改进点评工作台 · Replay Inspector")
        self.root.configure(bg=BG)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = max(0, (screen_w - WIN_W) // 2)
        y = max(0, (screen_h - WIN_H) // 2)
        self.root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
        self.root.resizable(False, False)

        self.storage = ReviewStorage()
        self.session: Optional[ReplaySession] = None
        self.current_ply: int = 0
        self.is_playing: bool = False
        self.play_speed_ms: int = 1200
        self.weights = EvalWeights()

        # 点选推荐走法缓存
        self.selected_rec_from: Optional[tuple] = None
        self.selected_rec_to: Optional[tuple] = None

        # 棋盘与控制区
        left_frame = tk.Frame(root, bg=BG)
        left_frame.pack(side="left", padx=8, pady=6)

        self.canvas = tk.Canvas(left_frame, width=BOARD_W, height=BOARD_H, bg=BG, highlightthickness=0)
        self.canvas.pack(side="top")
        self.canvas.bind("<Button-1>", self._on_board_click)

        self._build_player_controls(left_frame)

        # 加载棋子切片
        self._load_sprites()

        # 右侧面板
        self._build_panel()

        # 绑定全局快捷键
        self._bind_hotkeys()

        # 初始文件载入
        if initial_file and os.path.exists(initial_file):
            self.load_replay_file(initial_file)
        else:
            # 优先自动载入 games 目录最新一局
            recent_games = sorted(glob.glob("games/game_*.json"), key=os.path.getmtime, reverse=True)
            if recent_games:
                self.load_replay_file(recent_games[0])
            else:
                sav_files = sorted(glob.glob("军旗复盘/*.sav"))
                if sav_files:
                    self.load_replay_file(sav_files[0])

    def _load_sprites(self):
        """加载 APK 原版棋子贴图。"""
        self.sprites = {}
        self.sprite_back = None
        if not os.path.exists(CELLS_DIR):
            return
        for f in os.listdir(CELLS_DIR):
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

    def _build_player_controls(self, parent: tk.Frame):
        """构建棋盘下方的步进播放控制条。"""
        ctrl_frame = tk.Frame(parent, bg=BG)
        ctrl_frame.pack(side="top", fill="x", pady=(4, 0))

        # 按钮组
        btn_box = tk.Frame(ctrl_frame, bg=BG)
        btn_box.pack(side="top", pady=2)

        tk.Button(btn_box, text="|<< 开局", font=FONT_S, width=7, command=self.goto_start).pack(side="left", padx=2)
        tk.Button(btn_box, text="◀ 上一手", font=FONT_S, width=8, command=self.prev_ply).pack(side="left", padx=2)
        self.btn_play = tk.Button(btn_box, text="▶ 自动播放", font=FONT_S, width=9, bg="#E8F8F5", command=self.toggle_play)
        self.btn_play.pack(side="left", padx=2)
        tk.Button(btn_box, text="下一手 ▶", font=FONT_S, width=8, command=self.next_ply).pack(side="left", padx=2)
        tk.Button(btn_box, text="终局 >>|", font=FONT_S, width=7, command=self.goto_end).pack(side="left", padx=2)

        # 进度滑块
        slider_box = tk.Frame(ctrl_frame, bg=BG)
        slider_box.pack(side="top", fill="x", pady=2)

        self.ply_var = tk.IntVar(value=0)
        self.slider = tk.Scale(slider_box, variable=self.ply_var, from_=0, to=0, orient="horizontal",
                               showvalue=False, bg=BG, highlightthickness=0, command=self._on_slider_move)
        self.slider.pack(side="left", fill="x", expand=True, padx=4)

        self.lbl_ply_progress = tk.Label(slider_box, text="0 / 0", font=FONT_S, bg=BG, width=10)
        self.lbl_ply_progress.pack(side="left")

    def _build_panel(self):
        """构建右侧综合控制与点评面板。"""
        p = tk.Frame(self.root, bg=BG, width=PANEL_W)
        p.pack(side="left", fill="both", expand=True, padx=(4, 12), pady=6)

        # 1. 顶部标题与对局加载
        top_box = tk.LabelFrame(p, text="复盘文件管理", font=FONT_S, bg=BG, fg="#222", padx=6, pady=4)
        top_box.pack(fill="x", pady=2)

        row_f = tk.Frame(top_box, bg=BG)
        row_f.pack(fill="x", pady=2)
        tk.Button(row_f, text="📂 打开 .sav 复盘", font=FONT_S, command=self.open_sav_file).pack(side="left", padx=2, expand=True, fill="x")
        tk.Button(row_f, text="🎮 打开对战 .json", font=FONT_S, command=self.open_json_file).pack(side="left", padx=2, expand=True, fill="x")

        # 最近对战快速下拉
        row_recent = tk.Frame(top_box, bg=BG)
        row_recent.pack(fill="x", pady=2)
        tk.Label(row_recent, text="最近对局:", font=FONT_S, bg=BG).pack(side="left")
        self.recent_combo = ttk.Combobox(row_recent, font=FONT_S, state="readonly", width=32)
        self.recent_combo.pack(side="left", padx=4, fill="x", expand=True)
        self.recent_combo.bind("<<ComboboxSelected>>", self._on_recent_selected)
        self._refresh_recent_list()

        # 2. 对局信息与当前步着法
        info_box = tk.LabelFrame(p, text="对局态势与当前手", font=FONT_S, bg=BG, fg="#222", padx=6, pady=3)
        info_box.pack(fill="x", pady=3)

        self.lbl_game_meta = tk.Label(info_box, text="文件: -", font=FONT_S, bg=BG, anchor="w", fg="#444")
        self.lbl_game_meta.pack(fill="x")

        self.lbl_current_move = tk.Label(info_box, text="当前着法: -", font=("Microsoft YaHei", 11, "bold"),
                                         bg=BG, anchor="w", fg="#0B5394")
        self.lbl_current_move.pack(fill="x", pady=1)

        # 3. AI 智能研判区
        ai_box = tk.LabelFrame(p, text="AI 智能研判与对比", font=FONT_S, bg=BG, fg="#222", padx=6, pady=3)
        ai_box.pack(fill="x", pady=2)

        ai_btn_row = tk.Frame(ai_box, bg=BG)
        ai_btn_row.pack(fill="x", pady=1)
        self.btn_eval_ai = tk.Button(ai_btn_row, text="⚡ 计算当前局面 AI 研判 (Top-3 推荐与胜率)", font=FONT_S,
                                     bg="#FEF9E7", fg="#7D6608", command=self.eval_current_position)
        self.btn_eval_ai.pack(side="left", fill="x", expand=True)

        self.lbl_ai_eval = tk.Label(ai_box, text="AI 推荐: 点击上方按钮即时推演", font=FONT_S, bg=BG, anchor="w", fg="#555")
        self.lbl_ai_eval.pack(fill="x", pady=1)

        self.lbl_ai_winrate = tk.Label(ai_box, text="胜率预测: -", font=FONT_S, bg=BG, anchor="w", fg="#555")
        self.lbl_ai_winrate.pack(fill="x")

        # 4. 单步点评与纠偏录入区
        review_box = tk.LabelFrame(p, text="单步战术点评与纠偏 (本步批注)", font=FONT_S, bg=BG, fg="#900C3F", padx=6, pady=3)
        review_box.pack(fill="x", pady=3)

        self.lbl_review_status = tk.Label(review_box, text="本步点评状态: 待批注", font=("Microsoft YaHei", 10, "bold"),
                                          bg=BG, fg="#777")
        self.lbl_review_status.pack(anchor="w")

        # 标签选择下拉/单选
        lbl_select_row = tk.Frame(review_box, bg=BG)
        lbl_select_row.pack(fill="x", pady=2)
        tk.Label(lbl_select_row, text="定性评级:", font=FONT_S, bg=BG).pack(side="left")

        self.label_var = tk.StringVar(value="expert_blunder")
        self.label_combo = ttk.Combobox(lbl_select_row, textvariable=self.label_var, font=FONT_S,
                                        state="readonly", width=28)
        self.label_combo["values"] = [f"{k} · {v}" for k, v in REVIEW_LABELS.items()]
        self.label_combo.current(0)
        self.label_combo.pack(side="left", padx=4, fill="x", expand=True)

        # 纠偏走法录入
        rec_row = tk.Frame(review_box, bg=BG)
        rec_row.pack(fill="x", pady=2)
        tk.Label(rec_row, text="推荐正着:", font=FONT_S, bg=BG).pack(side="left")
        self.rec_action_var = tk.StringVar(value="")
        self.entry_rec = tk.Entry(rec_row, textvariable=self.rec_action_var, font=FONT_S, width=18)
        self.entry_rec.pack(side="left", padx=4)
        tk.Button(rec_row, text="清空正着", font=FONT_S, command=self._clear_rec_action).pack(side="left", padx=2)
        tk.Label(rec_row, text="(或在棋盘点选)", font=FONT_S, bg=BG, fg="#666").pack(side="left")

        # 点评分析输入框
        tk.Label(review_box, text="战术心得与点评分析:", font=FONT_S, bg=BG, fg="#444").pack(anchor="w", pady=(2, 0))
        self.comment_text = tk.Text(review_box, height=4, width=44, font=FONT_S, bg="#FFF", relief="solid", bd=1)
        self.comment_text.pack(fill="x", pady=2)

        # 操作按钮
        act_btn_row = tk.Frame(review_box, bg=BG)
        act_btn_row.pack(fill="x", pady=3)
        tk.Button(act_btn_row, text="💾 保存/更新本步点评", font=FONT, bg="#E8F8F5", fg="#1E8449",
                  command=self.save_current_review).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(act_btn_row, text="🗑️ 删除本步点评", font=FONT_S, bg="#FDEDEC", fg="#922B21",
                  command=self.delete_current_review).pack(side="left", padx=2)

        # 5. 全量汇总与算法改进导出
        summary_box = tk.LabelFrame(p, text="全量复盘点评汇总与算法改进", font=FONT_S, bg=BG, fg="#1F618D", padx=6, pady=4)
        summary_box.pack(fill="x", pady=4)

        self.lbl_global_stat = tk.Label(summary_box, text="统计: 本局已批注 0 步  |  全库已收录 0 条点评",
                                        font=FONT_S, bg=BG, fg="#333")
        self.lbl_global_stat.pack(pady=2)

        tk.Button(summary_box, text="📊 一键汇总所有点评 (导出诊断报告与训练数据集)", font=FONT,
                  bg="#EBF5FB", fg="#1B4F72", command=self.export_all_reviews).pack(fill="x", pady=2)

    def _bind_hotkeys(self):
        """快捷键绑定。"""
        self.root.bind("<Left>", lambda e: self.prev_ply())
        self.root.bind("<Right>", lambda e: self.next_ply())
        self.root.bind("<Home>", lambda e: self.goto_start())
        self.root.bind("<End>", lambda e: self.goto_end())
        self.root.bind("<space>", lambda e: self.toggle_play())

    def _refresh_recent_list(self):
        """刷新最近对战对局列表。"""
        files = sorted(glob.glob("games/game_*.json"), key=os.path.getmtime, reverse=True)[:15]
        display_names = [os.path.basename(f) for f in files]
        self.recent_combo["values"] = display_names
        if display_names:
            self.recent_combo.current(0)

    def _on_recent_selected(self, event=None):
        sel = self.recent_combo.get()
        if sel:
            p = os.path.join("games", sel)
            if os.path.exists(p):
                self.load_replay_file(p)

    # ------------------------------------------------------------- 复盘加载与步进
    def open_sav_file(self):
        f = filedialog.askopenfilename(
            title="选择军棋原生 .sav 复盘文件",
            filetypes=[("军棋原生复盘", "*.sav"), ("所有文件", "*.*")]
        )
        if f:
            self.load_replay_file(f)

    def open_json_file(self):
        f = filedialog.askopenfilename(
            title="选择人机对战 JSON 记录文件",
            initialdir="games",
            filetypes=[("对战记录 JSON", "*.json"), ("所有文件", "*.*")]
        )
        if f:
            self.load_replay_file(f)

    def load_replay_file(self, file_path: str):
        """加载复盘并初始化步进器。"""
        try:
            self.session = ReplayManager.load_game(file_path)
            self.current_ply = 0
            self.slider.config(to=self.session.total_plies)
            self.ply_var.set(0)

            # 更新对局元数据
            meta = self.session.metadata
            base = os.path.basename(file_path)
            win_str = f"胜者: 座位{meta.get('winner')}" if meta.get('winner') is not None else "对局和棋或进行中"
            self.lbl_game_meta.config(text=f"文件: {base}  |  总手数: {self.session.total_plies}  |  {win_str}")

            self._update_display()
        except Exception as e:
            messagebox.showerror("加载失败", f"无法解析复盘文件:\n{file_path}\n错误信息: {e}")

    def goto_start(self):
        if not self.session:
            return
        self.current_ply = 0
        self.ply_var.set(0)
        self._update_display()

    def goto_end(self):
        if not self.session:
            return
        self.current_ply = self.session.total_plies
        self.ply_var.set(self.current_ply)
        self._update_display()

    def prev_ply(self):
        if not self.session or self.current_ply <= 0:
            return
        self.current_ply -= 1
        self.ply_var.set(self.current_ply)
        self._update_display()

    def next_ply(self):
        if not self.session or self.current_ply >= self.session.total_plies:
            if self.is_playing:
                self.toggle_play()
            return
        self.current_ply += 1
        self.ply_var.set(self.current_ply)
        self._update_display()

    def _on_slider_move(self, val):
        if not self.session:
            return
        self.current_ply = int(val)
        self._update_display()

    def toggle_play(self):
        if not self.session:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ 暂停播放", bg="#FADBD8")
            self._play_loop()
        else:
            self.btn_play.config(text="▶ 自动播放", bg="#E8F8F5")

    def _play_loop(self):
        if not self.is_playing:
            return
        if self.current_ply < self.session.total_plies:
            self.next_ply()
            self.root.after(self.play_speed_ms, self._play_loop)
        else:
            self.is_playing = False
            self.btn_play.config(text="▶ 自动播放", bg="#E8F8F5")

    # ------------------------------------------------------------- 棋盘绘制与交互
    def _update_display(self):
        """刷新棋盘界面与右侧信息。"""
        if not self.session:
            return

        total = self.session.total_plies
        self.lbl_ply_progress.config(text=f"{self.current_ply} / {total}")

        # 获取当步对应的盘面与动作
        if self.current_ply == 0:
            st = self.session.initial_state
            step = None
            self.lbl_current_move.config(text="第 0 手：初始盘面 (对局准备开局)")
        else:
            step = self.session.get_step(self.current_ply - 1)
            st = step.state_after
            clr = "红方" if step.turn_color == "r" else "蓝方"
            self.lbl_current_move.config(text=f"第 {step.ply + 1} 手 [{clr}]: {step.action_desc}")

        self._draw_board(st, step)
        self._load_review_for_current_ply()
        self._update_stats()

    def _draw_board(self, st: GameState, last_step: Optional[ReplayStep]):
        self.canvas.delete("all")

        # 1. 棋盘底色
        self.canvas.create_rectangle(MX, TITLE_H + MY, BOARD_W - MX, BOARD_H - MY, fill=WOOD, outline="")
        self.canvas.create_rectangle(MX, TITLE_H + MY + ROWS // 2 * CELL_H,
                                     BOARD_W - MX, TITLE_H + MY + ROWS // 2 * CELL_H + 34,
                                     fill=BG, outline="")

        # 2. 铁轨与公路线
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = rc_to_xy(r, c)
                for nr, nc in NEIGHBORS.get((r, c), []):
                    if (nr, nc) < (r, c):
                        continue
                    x2, y2 = rc_to_xy(nr, nc)
                    is_r = is_rail((r, c)) and is_rail((nr, nc)) and (nr, nc) in rail_neighbors((r, c))
                    if is_r:
                        self.canvas.create_line(x1, y1, x2, y2, fill=RAIL_FILL, width=5)
                        self.canvas.create_line(x1, y1, x2, y2, fill="white", width=2, dash=(4, 4))
                    else:
                        self.canvas.create_line(x1, y1, x2, y2, fill=ROAD_FILL, width=2)

        # 3. 行营与大本营特殊几何标记
        for cp in CAMPS:
            cx, cy = rc_to_xy(*cp)
            self.canvas.create_oval(cx - 24, cy - 24, cx + 24, cy + 24, fill=WOOD_DARK, outline="#8B5E34", width=2)
            self.canvas.create_text(cx, cy, text="营", font=FONT_S, fill="#5E3A1A")

        for hq in HQS:
            hx, hy = rc_to_xy(*hq)
            self.canvas.create_rectangle(hx - 28, hy - 18, hx + 28, hy + 18, fill="#C49A45", outline="#784212", width=2)
            self.canvas.create_text(hx, hy, text="大本营", font=FONT_S, fill="#5E3A1A")

        # 4. 绘制棋子
        for (r, c), p in st.board.items():
            cx, cy = rc_to_xy(r, c)
            w_half, h_half = CELL_W // 2 - 3, CELL_H // 2 - 3
            if not p.revealed:
                if self.sprite_back:
                    self.canvas.create_image(cx, cy, image=self.sprite_back)
                else:
                    self.canvas.create_rectangle(cx - w_half, cy - h_half, cx + w_half, cy + h_half,
                                                 fill=BACK_FILL, outline=BACK_EDGE, width=2)
                    self.canvas.create_text(cx, cy, text="暗", font=FONT_PIECE, fill="white")
            else:
                sp = self.sprites.get((p.color, p.rank))
                if sp:
                    self.canvas.create_image(cx, cy, image=sp)
                else:
                    fill_c = face_color(p)
                    self.canvas.create_rectangle(cx - w_half, cy - h_half, cx + w_half, cy + h_half,
                                                 fill=fill_c, outline="#222", width=2)
                    self.canvas.create_text(cx, cy, text=piece_label(p), font=FONT_PIECE, fill=FACE_TEXT)

        # 5. 上一手动作轨迹高亮
        if last_step:
            act = last_step.action
            if act.kind == "flip":
                fx, fy = rc_to_xy(*act.frm)
                self.canvas.create_rectangle(fx - CELL_W//2 + 2, fy - CELL_H//2 + 2,
                                             fx + CELL_W//2 - 2, fy + CELL_H//2 - 2,
                                             outline=PLAYED_COLOR, width=3)
            elif act.kind == "move":
                fx, fy = rc_to_xy(*act.frm)
                tx, ty = rc_to_xy(*act.to)
                self.canvas.create_rectangle(fx - CELL_W//2 + 2, fy - CELL_H//2 + 2,
                                             fx + CELL_W//2 - 2, fy + CELL_H//2 - 2,
                                             outline=SEL, width=2, dash=(3, 3))
                self.canvas.create_rectangle(tx - CELL_W//2 + 2, ty - CELL_H//2 + 2,
                                             tx + CELL_W//2 - 2, ty + CELL_H//2 - 2,
                                             outline=PLAYED_COLOR, width=3)
                self.canvas.create_line(fx, fy, tx, ty, fill=PLAYED_COLOR, width=3, arrow="last")

        # 6. 点选推荐正着高亮 (若有)
        if self.selected_rec_from:
            rx, ry = rc_to_xy(*self.selected_rec_from)
            self.canvas.create_rectangle(rx - CELL_W//2 + 1, ry - CELL_H//2 + 1,
                                         rx + CELL_W//2 - 1, ry + CELL_H//2 - 1,
                                         outline=REC_COLOR, width=3)
        if self.selected_rec_to:
            rx, ry = rc_to_xy(*self.selected_rec_to)
            self.canvas.create_rectangle(rx - CELL_W//2 + 1, ry - CELL_H//2 + 1,
                                         rx + CELL_W//2 - 1, ry + CELL_H//2 - 1,
                                         outline=REC_COLOR, width=3)

    def _on_board_click(self, event):
        """点击棋盘格子快速设定推荐正着。"""
        rc = xy_to_rc(event.x, event.y)
        if not rc:
            return

        if self.selected_rec_from is None:
            self.selected_rec_from = rc
            self.rec_action_var.set(f"从 {rc}")
        elif self.selected_rec_to is None:
            if rc == self.selected_rec_from:
                # 点击同一格视为翻棋推荐
                self.rec_action_var.set(f"翻 {rc}")
                self.selected_rec_from = rc
                self.selected_rec_to = None
            else:
                self.selected_rec_to = rc
                self.rec_action_var.set(f"走 {self.selected_rec_from}->{self.selected_rec_to}")
        else:
            self.selected_rec_from = rc
            self.selected_rec_to = None
            self.rec_action_var.set(f"从 {rc}")

        step = self.session.get_step(self.current_ply - 1) if self.session and self.current_ply > 0 else None
        st = step.state_after if step else (self.session.initial_state if self.session else None)
        if st:
            self._draw_board(st, step)

    def _clear_rec_action(self):
        self.selected_rec_from = None
        self.selected_rec_to = None
        self.rec_action_var.set("")
        self._update_display()

    # ------------------------------------------------------------- AI 研判
    def eval_current_position(self):
        """异步调用专家引擎与胜率模型研判当前盘面。"""
        if not self.session:
            return

        step = self.session.get_step(self.current_ply - 1) if self.current_ply > 0 else None
        st = step.state_after if step else self.session.initial_state

        if st.is_terminal():
            self.lbl_ai_eval.config(text="当前盘面已终局，无需研判。")
            return

        self.lbl_ai_eval.config(text="正在进行专家深搜推演中…")
        self.lbl_ai_winrate.config(text="胜率预测计算中…")

        def _work():
            try:
                from .ai import ExpertAgent
                agent = ExpertAgent(search=SearchConfig(depth=3, time_limit_ms=1500), weights=self.weights, seed=42)
                top_scored = agent.choose_actions(st, topn=3)

                tip_str = " | ".join([f"{a} ({s:+.1f})" for a, s in top_scored]) if top_scored else "无可用走法"

                # 胜率预测
                p_red, p_draw, p_blue = 50.0, 0.0, 50.0
                try:
                    from .hybrid_engine import HybridDecisionEngine
                    mp = "models/best.pt" if os.path.exists("models/best.pt") else "models/bc_best.pt"
                    engine = HybridDecisionEngine(model_path=mp, k_worlds=4)
                    info = engine.evaluate_position(st)
                    p_win, p_draw_val, p_loss = info["win"], info["draw"], info["loss"]
                    if st.my_color() == "r":
                        p_red, p_blue = p_win * 100.0, p_loss * 100.0
                    else:
                        p_red, p_blue = p_loss * 100.0, p_win * 100.0
                    p_draw = p_draw_val * 100.0
                except Exception:
                    pass

                self.root.after(0, lambda: self.lbl_ai_eval.config(text=f"AI Top3: {tip_str}"))
                self.root.after(0, lambda: self.lbl_ai_winrate.config(
                    text=f"预测胜率: 红胜 {p_red:.1f}%  |  和棋 {p_draw:.1f}%  |  蓝胜 {p_blue:.1f}%"
                ))
            except Exception as e:
                self.root.after(0, lambda: self.lbl_ai_eval.config(text=f"推演失败: {e}"))

        threading.Thread(target=_work, daemon=True).start()

    # ------------------------------------------------------------- 点评保存与管理
    def _load_review_for_current_ply(self):
        """读取并展示当前步的历史点评。"""
        if not self.session:
            return

        source = self.session.file_path
        rev = self.storage.get_review(source, self.current_ply)

        self.comment_text.delete("1.0", "end")
        if rev:
            lbl_name = rev.get("label_name", rev.get("label", ""))
            self.lbl_review_status.config(text=f"本步点评: 【{lbl_name}】 (已批注)", fg="#C0392B")
            self.comment_text.insert("1.0", rev.get("comment", ""))

            # 选定 combo
            lbl_key = rev.get("label", "expert_blunder")
            for idx, item in enumerate(self.label_combo["values"]):
                if item.startswith(lbl_key):
                    self.label_combo.current(idx)
                    break

            rec_desc = rev.get("action_recommended_desc", "")
            self.rec_action_var.set(rec_desc)
        else:
            self.lbl_review_status.config(text="本步点评状态: 待批注", fg="#777")
            self.rec_action_var.set("")

    def save_current_review(self):
        """保存或更新当前步点评。"""
        if not self.session:
            messagebox.showwarning("提示", "请先打开复盘文件！")
            return

        source = self.session.file_path
        step = self.session.get_step(self.current_ply - 1) if self.current_ply > 0 else None
        st = step.state_before if step else self.session.initial_state
        action_played = step.action if step else Action("pass")
        action_desc = step.action_desc if step else "开局初始盘面"

        # 解析标签
        raw_val = self.label_combo.get().split(" · ")[0].strip()
        label_key = raw_val if raw_val in REVIEW_LABELS else "expert_blunder"

        comment = self.comment_text.get("1.0", "end").strip()

        # 解析推荐走法
        rec_action = None
        if self.selected_rec_from:
            if self.selected_rec_to:
                rec_action = Action("move", self.selected_rec_from, self.selected_rec_to)
            else:
                rec_action = Action("flip", self.selected_rec_from)

        self.storage.save_review(
            source_file=source,
            ply=self.current_ply,
            state=st,
            action_played=action_played,
            label=label_key,
            comment=comment,
            action_recommended=rec_action,
            action_played_desc=action_desc,
            action_recommended_desc=self.rec_action_var.get().strip(),
            metadata={"total_plies": self.session.total_plies},
        )

        messagebox.showinfo("成功", f"第 {self.current_ply} 步战术点评已保存！")
        self._load_review_for_current_ply()
        self._update_stats()

    def delete_current_review(self):
        """删除当前步点评。"""
        if not self.session:
            return
        source = self.session.file_path
        if self.storage.delete_review(source, self.current_ply):
            messagebox.showinfo("成功", f"已删除第 {self.current_ply} 步点评记录。")
            self._load_review_for_current_ply()
            self._update_stats()
        else:
            messagebox.showinfo("提示", "当前步尚无点评记录。")

    def _update_stats(self):
        """刷新统计计数。"""
        if not self.session:
            return
        src = self.session.file_path
        cur_cnt = self.storage.count(src)
        tot_cnt = self.storage.count()
        self.lbl_global_stat.config(text=f"统计: 本局已批注 {cur_cnt} 步  |  全库已收录 {tot_cnt} 条点评")

    # ------------------------------------------------------------- 一键汇总与导出
    def export_all_reviews(self):
        """汇总全量点评并输出算法改进报告与训练数据集。"""
        pipeline = ReviewSummaryPipeline(self.storage)
        res = pipeline.run_all()

        if res["total_reviews"] == 0:
            messagebox.showinfo("提示", "当前点评库为空，暂无点评记录可汇总。请先对关键手进行点评批注！")
            return

        msg = (
            f"✅ 全量复盘点评汇总完成！\n\n"
            f"• 总点评记录: {res['total_reviews']} 条\n"
            f"• 诊断报告: {res['report_file']}\n"
            f"• 算法改进训练集: {res['dataset_file']} ({res['dataset_samples']} 条样本)\n"
            f"• 自动化回归测试: {res['test_file']} ({res['tests_generated']} 个用例)\n\n"
            f"您可以直接运行以下命令执行模型微调蒸馏与回归测试：\n"
            f"python -m junqi train_value_distill --data {res['dataset_file']}"
        )
        messagebox.showinfo("全量汇总导出成功", msg)


def run_replay_gui(initial_file: Optional[str] = None):
    root = tk.Tk()
    app = ReplayViewerApp(root, initial_file=initial_file)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="军棋对局复盘与点评工作台")
    parser.add_argument("--file", "-f", default=None, help="初始载入的对局文件 (.sav 或 .json)")
    args = parser.parse_args()
    run_replay_gui(args.file)
