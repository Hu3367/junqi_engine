"""专用交互式打标与抽查核验 GUI 工作台。

功能说明：
1. 真实复现抽取自人类实战复盘的中后盘交火局面；
2. 直观对比【人类玩家当时走法】与【专家搜索引擎推荐走法 + 评分】；
3. 支持人工抽查与一键重新打标修正（胜/和/负/存疑），自动记录人工核验状态；
4. 支持按“人类与专家分歧局”、“未核验局”等条件高效过滤筛选；
5. 即时增量保存核验结果至 JSON 数据集。

使用方法：
    python -m junqi label_gui
    python -m junqi.label_gui --data datasets/distill_tactical_labeled.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import messagebox
from typing import Dict, List, Optional

from .config import RuleConfig
from .gui import (ALL_POSITIONS, BACK_EDGE, BACK_FILL, BG, BOARD_H, BOARD_W,
                  B_FACE, CAMPS, CELLS_DIR, CELL_H, CELL_W, COLOR_CN, COLS,
                  FACE_TEXT, FONT, FONT_PIECE, FONT_S, FONT_TITLE, HQS,
                  NEIGHBORS, RAIL_FILL, RANK_CN, ROAD_FILL, ROWS, R_FACE,
                  TITLE_H, WOOD, WOOD_DARK, face_color, is_rail, piece_label,
                  rail_neighbors, rc_to_xy, xy_to_rc)
from .rules import Rank, other
from .state import Action, GameState, Piece
from .tactical_sampler import action_to_dict, dict_to_action, dict_to_state

# 界面专用配色与尺寸
PANEL_W = 410
WIN_W = BOARD_W + PANEL_W + 24
WIN_H = BOARD_H + 16

HUMAN_MOVE_COLOR = "#00B4D8"   # 青蓝色标注人类走法
EXPERT_MOVE_COLOR = "#FF9E00"  # 亮金黄标注专家推荐走法
MATCH_COLOR = "#2B9348"        # 意见一致绿色
DIFF_COLOR = "#D90429"         # 意见分歧红色

# 招法合理性裁决体系定义 (Action Judgment System)
JUDGMENTS = {
    "expert_better": ("⭐ 专家更优 (采纳推荐)", "#27AE60", "#E8F8F5"),
    "human_better": ("👤 人类更佳 (专家短视)", "#2980B9", "#EBF5FB"),
    "both_fine": ("🤝 两手皆可 (常规好棋)", "#16A085", "#E8F6F3"),
    "expert_blunder": ("❌ 专家恶手 (战术硬伤)", "#C0392B", "#FDEDEC"),
    "skip": ("❓ 标记存疑/跳过", "#8E44AD", "#F4ECF7"),
}



class LabelInspectorApp:
    def __init__(self, root: tk.Tk, data_path: str = "datasets/distill_tactical_labeled.json"):
        self.root = root
        self.root.title("军棋实战交火打标与核验工作台 · Label Inspector")
        self.root.configure(bg=BG)

        # 屏幕居中
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = max(0, (screen_w - WIN_W) // 2)
        y = max(0, (screen_h - WIN_H) // 2)
        self.root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
        self.root.resizable(False, False)

        self.data_path = data_path
        self.cfg = RuleConfig()
        self.dataset: List[dict] = []
        self.filtered_indices: List[int] = []
        self.current_idx_in_filter = 0

        # 过滤筛选变量
        self.filter_divergence_only = tk.BooleanVar(value=False)
        self.filter_unverified_only = tk.BooleanVar(value=False)
        self.is_searching = False

        # 棋盘画布
        self.canvas = tk.Canvas(root, width=BOARD_W, height=BOARD_H, bg=BG, highlightthickness=0)
        self.canvas.pack(side="left", padx=6, pady=6)

        self._load_sprites()
        self._build_panel()
        self._bind_hotkeys()

        # 加载数据
        self.load_data(data_path)

    def _load_sprites(self):
        """加载 APK 原版棋子切片。"""
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

    # ------------------------------------------------------------- 右侧控制面板

    def _build_panel(self):
        p = tk.Frame(self.root, bg=BG, width=PANEL_W)
        p.pack(side="left", fill="both", expand=True, padx=(4, 12), pady=8)

        # 标题
        tk.Label(p, text="实战交火核验工作台", font=FONT_TITLE, bg=BG, fg="#1F4E1D").pack(pady=(0, 4))

        # 1. 导航条 (跳转与进度)
        nav_box = tk.LabelFrame(p, text="局面定位与浏览", font=FONT_S, bg=BG, fg="#333", padx=8, pady=4)
        nav_box.pack(fill="x", pady=2)

        nav_row1 = tk.Frame(nav_box, bg=BG)
        nav_row1.pack(fill="x", pady=2)

        self.idx_label = tk.Label(nav_row1, text="编号: 0 / 0", font=FONT, bg=BG, fg="#222")
        self.idx_label.pack(side="left")

        tk.Label(nav_row1, text="跳至#", font=FONT_S, bg=BG).pack(side="left", padx=(10, 2))
        self.jump_var = tk.StringVar(value="1")
        tk.Entry(nav_row1, textvariable=self.jump_var, width=6, font=FONT_S).pack(side="left")
        tk.Button(nav_row1, text="跳转", font=FONT_S, command=self.jump_to_id).pack(side="left", padx=4)

        nav_row2 = tk.Frame(nav_box, bg=BG)
        nav_row2.pack(fill="x", pady=4)
        tk.Button(nav_row2, text="◀ 上一个 (←)", font=FONT_S, width=11, command=self.prev_item).pack(side="left", padx=2)
        tk.Button(nav_row2, text="下一个 (→) ▶", font=FONT_S, width=11, command=self.next_item).pack(side="left", padx=2)
        tk.Button(nav_row2, text="🎲 随机 (R)", font=FONT_S, width=9, command=self.random_item).pack(side="left", padx=2)

        # 过滤复选框
        filter_row = tk.Frame(nav_box, bg=BG)
        filter_row.pack(fill="x", pady=2)
        tk.Checkbutton(filter_row, text="仅看分歧局 (人机不同)", variable=self.filter_divergence_only,
                       bg=BG, font=FONT_S, command=self.apply_filter).pack(side="left", padx=(0, 6))
        tk.Checkbutton(filter_row, text="仅看未核验", variable=self.filter_unverified_only,
                       bg=BG, font=FONT_S, command=self.apply_filter).pack(side="left")

        # 2. 局面态势与元数据
        info_box = tk.LabelFrame(p, text="实战来源与战场态势", font=FONT_S, bg=BG, fg="#333", padx=8, pady=4)
        info_box.pack(fill="x", pady=4)

        self.lbl_game = tk.Label(info_box, text="复盘: -", font=FONT_S, bg=BG, fg="#444", anchor="w")
        self.lbl_game.pack(fill="x")
        self.lbl_turn = tk.Label(info_box, text="行动: -", font=FONT, bg=BG, fg="#111", anchor="w")
        self.lbl_turn.pack(fill="x")
        self.lbl_material = tk.Label(info_box, text="子力: -", font=FONT_S, bg=BG, fg="#444", anchor="w")
        self.lbl_material.pack(fill="x")

        # 3. 走法与战术推演对比 (人类走法 vs 专家推荐)
        cmp_box = tk.LabelFrame(p, text="招法与战术推演对比", font=FONT_S, bg=BG, fg="#333", padx=8, pady=4)
        cmp_box.pack(fill="x", pady=3)

        self.lbl_human_move = tk.Label(cmp_box, text="人类实战: -", font=FONT, bg=BG, fg=HUMAN_MOVE_COLOR, anchor="w")
        self.lbl_human_move.pack(fill="x")

        self.lbl_expert_move = tk.Label(cmp_box, text="专家推荐: -", font=FONT, bg=BG, fg="#B25900", anchor="w")
        self.lbl_expert_move.pack(fill="x")

        self.lbl_match_status = tk.Label(cmp_box, text="吻合状态: -", font=FONT, bg=BG, fg="#333", anchor="w")
        self.lbl_match_status.pack(fill="x")

        self.lbl_pos_adv = tk.Label(cmp_box, text="盘面态势参考: -", font=FONT_S, bg=BG, fg="#555", anchor="w")
        self.lbl_pos_adv.pack(fill="x")

        self.lbl_expert_score = tk.Label(cmp_box, text="专家评分: -", font=FONT_S, bg=BG, fg="#666", anchor="w")
        self.lbl_expert_score.pack(fill="x")

        # 选项 B: 5 层超深度推演按钮与状态展示
        d5_row = tk.Frame(cmp_box, bg=BG)
        d5_row.pack(fill="x", pady=(3, 2))
        self.btn_depth5 = tk.Button(d5_row, text="⚡ 启动 5 层超深度推演 (Depth 5)", font=FONT_S,
                                    bg="#FEF9E7", fg="#B7950B", activebackground="#FCF3CF",
                                    command=self.start_depth5_search)
        self.btn_depth5.pack(side="left", fill="x", expand=True)

        self.lbl_depth5_status = tk.Label(cmp_box, text="", font=FONT_S, bg=BG, fg="#888", anchor="w")
        self.lbl_depth5_status.pack(fill="x")

        # 4. 人工招法合理性裁决判定区 (Action Judgment)
        label_box = tk.LabelFrame(p, text="人工招法合理性裁决 (快捷键: 1/2/3/4/S)", font=FONT_S, bg=BG,
                                  fg="#900C3F", padx=8, pady=4)
        label_box.pack(fill="x", pady=3)

        self.lbl_current_label = tk.Label(label_box, text="当前裁决: 待审定", font=("Microsoft YaHei", 11, "bold"),
                                          bg=BG, fg="#555")
        self.lbl_current_label.pack(pady=1)

        # 按钮行 1: 专家更优 vs 人类更佳
        btn_row1 = tk.Frame(label_box, bg=BG)
        btn_row1.pack(fill="x", pady=2)
        self.btn_expert = tk.Button(btn_row1, text="⭐ 专家更优 (1/E)", font=FONT, bg="#E8F8F5",
                                    activebackground="#A3E4D7", fg="#1E8449",
                                    command=lambda: self.set_judgment("expert_better"))
        self.btn_expert.pack(side="left", padx=2, expand=True, fill="x")

        self.btn_human = tk.Button(btn_row1, text="👤 人类更佳 (2/H)", font=FONT, bg="#EBF5FB",
                                   activebackground="#AED6F1", fg="#2471A3",
                                   command=lambda: self.set_judgment("human_better"))
        self.btn_human.pack(side="left", padx=2, expand=True, fill="x")

        # 按钮行 2: 两手皆可 vs 专家恶手
        btn_row2 = tk.Frame(label_box, bg=BG)
        btn_row2.pack(fill="x", pady=2)
        self.btn_both = tk.Button(btn_row2, text="🤝 两手皆可 (3/B)", font=FONT, bg="#E8F6F3",
                                  activebackground="#A2D9CE", fg="#117864",
                                  command=lambda: self.set_judgment("both_fine"))
        self.btn_both.pack(side="left", padx=2, expand=True, fill="x")

        self.btn_blunder = tk.Button(btn_row2, text="❌ 专家恶手 (4/X)", font=FONT, bg="#FDEDEC",
                                     activebackground="#FADBD8", fg="#922B21",
                                     command=lambda: self.set_judgment("expert_blunder"))
        self.btn_blunder.pack(side="left", padx=2, expand=True, fill="x")

        # 按钮行 3: 存疑/跳过
        btn_row3 = tk.Frame(label_box, bg=BG)
        btn_row3.pack(fill="x", pady=1)
        tk.Button(btn_row3, text="❓ 标记存疑/跳过 (0/S)", font=FONT_S, width=18, bg="#F4ECF7",
                  activebackground="#D7BDE2", fg="#6C3483",
                  command=lambda: self.set_judgment("skip")).pack(side="left", padx=2, expand=True, fill="x")

        # 备注输入
        comment_row = tk.Frame(label_box, bg=BG)
        comment_row.pack(fill="x", pady=(3, 1))
        tk.Label(comment_row, text="核验备注:", font=FONT_S, bg=BG).pack(side="left")
        self.comment_var = tk.StringVar()
        self.entry_comment = tk.Entry(comment_row, textvariable=self.comment_var, font=FONT_S)
        self.entry_comment.pack(side="left", fill="x", expand=True, padx=4)
        self.entry_comment.bind("<FocusOut>", lambda e: self.update_comment())
        self.entry_comment.bind("<Return>", lambda e: self.update_comment())

        # 5. 底部保存与统计
        bot_box = tk.Frame(p, bg=BG)
        bot_box.pack(fill="x", side="bottom", pady=4)

        tk.Button(bot_box, text="💾 保存已核验数据集 (Ctrl+S)", font=FONT, bg="#27AE60", fg="white",
                  activebackground="#1E8449", activeforeground="white",
                  command=self.save_data).pack(fill="x", pady=2)

        self.lbl_stats = tk.Label(bot_box, text="统计: -", font=FONT_S, bg=BG, fg="#555", justify="left")
        self.lbl_stats.pack(pady=2)

    def _bind_hotkeys(self):
        """绑定快捷键。"""
        self.root.bind("<Left>", lambda e: self.prev_item())
        self.root.bind("<Right>", lambda e: self.next_item())
        self.root.bind("<r>", lambda e: self.random_item())
        self.root.bind("<R>", lambda e: self.random_item())
        self.root.bind("<Control-s>", lambda e: self.save_data())
        self.root.bind("<Control-S>", lambda e: self.save_data())
        self.root.bind("<F5>", lambda e: self.start_depth5_search())

        # 裁决快捷键：1/E=专家更优, 2/H=人类更佳, 3/B=两手皆可, 4/X=专家恶手, 0/S=跳过
        self.root.bind("1", lambda e: self._handle_judgment_key("expert_better"))
        self.root.bind("e", lambda e: self._handle_judgment_key("expert_better"))
        self.root.bind("E", lambda e: self._handle_judgment_key("expert_better"))

        self.root.bind("2", lambda e: self._handle_judgment_key("human_better"))
        self.root.bind("h", lambda e: self._handle_judgment_key("human_better"))
        self.root.bind("H", lambda e: self._handle_judgment_key("human_better"))

        self.root.bind("3", lambda e: self._handle_judgment_key("both_fine"))
        self.root.bind("b", lambda e: self._handle_judgment_key("both_fine"))
        self.root.bind("B", lambda e: self._handle_judgment_key("both_fine"))

        self.root.bind("4", lambda e: self._handle_judgment_key("expert_blunder"))
        self.root.bind("x", lambda e: self._handle_judgment_key("expert_blunder"))
        self.root.bind("X", lambda e: self._handle_judgment_key("expert_blunder"))

        self.root.bind("0", lambda e: self._handle_judgment_key("skip"))
        self.root.bind("s", lambda e: self._handle_judgment_key("skip"))
        self.root.bind("S", lambda e: self._handle_judgment_key("skip"))

    def _handle_judgment_key(self, judgment: str):
        if self.root.focus_get() == self.entry_comment:
            return
        self.set_judgment(judgment)

    # ------------------------------------------------------------- 数据控制与渲染

    def load_data(self, path: str):
        if not os.path.exists(path):
            messagebox.showwarning("提示", f"未找到打标数据集文件:\n{path}\n请先生成或选择正确文件。")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
            self.data_path = path
            self.apply_filter()
            self.show_current()
        except Exception as e:
            messagebox.showerror("读取错误", f"加载数据失败: {e}")

    def apply_filter(self):
        """根据分歧/核验状态更新过滤索引列表。"""
        if not self.dataset:
            self.filtered_indices = []
            return

        div_only = self.filter_divergence_only.get()
        unver_only = self.filter_unverified_only.get()

        indices = []
        for i, item in enumerate(self.dataset):
            if div_only and item.get("human_expert_match", True):
                continue
            if unver_only and item.get("human_verified", False):
                continue
            indices.append(i)

        self.filtered_indices = indices
        self.current_idx_in_filter = 0
        self.show_current()

    def get_current_item(self) -> Optional[dict]:
        if not self.filtered_indices:
            return None
        actual_idx = self.filtered_indices[self.current_idx_in_filter]
        return self.dataset[actual_idx]

    def show_current(self):
        """渲染当前选中的交火局面。"""
        item = self.get_current_item()
        if item is None:
            self.canvas.delete("all")
            self.idx_label.config(text="无匹配局面")
            self.lbl_game.config(text="复盘: -")
            self.lbl_turn.config(text="行动: -")
            self.lbl_material.config(text="子力: -")
            self.lbl_human_move.config(text="人类实战: -")
            self.lbl_expert_move.config(text="专家推荐: -")
            self.lbl_match_status.config(text="吻合状态: -")
            self.lbl_pos_adv.config(text="盘面态势参考: -")
            self.lbl_expert_score.config(text="专家评分: -")
            self.lbl_depth5_status.config(text="")
            self.lbl_current_label.config(text="当前裁决: 待审定", fg="#777")
            self.comment_var.set("")
            self.update_stats()
            return

        st = dict_to_state(item["state_dict"], cfg=self.cfg)
        self.state = st

        # 1. 更新顶部索引与导航
        curr_num = self.current_idx_in_filter + 1
        total_num = len(self.filtered_indices)
        orig_id = item.get("id", actual_idx := self.filtered_indices[self.current_idx_in_filter])
        self.idx_label.config(text=f"当前: {curr_num} / {total_num} (原ID: #{orig_id})")
        self.jump_var.set(str(orig_id))

        # 2. 更新元数据
        phase_str = "中盘交火" if item.get("phase") == 1 else "残局鏖战"
        self.lbl_game.config(text=f"复盘: {item.get('game_file')} (第 {item.get('ply')} 步 · {phase_str})")

        my_c = st.my_color()
        turn_str = f"轮到 {'红方' if my_c == 'r' else '蓝方'} 行动 (Seat {st.turn})"
        self.lbl_turn.config(text=turn_str, fg=R_FACE if my_c == 'r' else B_FACE)

        # 统计子力对比
        my_alive = sum(1 for p in st.board.values() if p.revealed and p.color == my_c)
        opp_c = other(my_c) if my_c else "b"
        opp_alive = sum(1 for p in st.board.values() if p.revealed and p.color == opp_c)
        hidden_count = sum(1 for p in st.board.values() if not p.revealed)
        self.lbl_material.config(text=f"存活明子: 己 {my_alive} 枚 vs 敌 {opp_alive} 枚 | 暗子剩余: {hidden_count} 枚")

        # 3. 走法对比与状态
        h_act = dict_to_action(item.get("human_action"))
        e_act = dict_to_action(item.get("expert_action"))
        self.human_action = h_act
        self.expert_action = e_act

        h_desc = self.format_action(st, h_act)
        e_desc = self.format_action(st, e_act)
        self.lbl_human_move.config(text=f"人类实战走法:  {h_desc}")

        e_depth = item.get("expert_depth", 3)
        depth_tag = f" [Depth {e_depth}超算]" if e_depth >= 5 else ""
        self.lbl_expert_move.config(text=f"专家推荐走法:  {e_desc}{depth_tag}")

        is_match = item.get("human_expert_match", (h_act == e_act))
        if is_match:
            self.lbl_match_status.config(text="吻合状态:  ✅ 人机判断一致 (Agreement)", fg=MATCH_COLOR)
        else:
            self.lbl_match_status.config(text="吻合状态:  ⚠️ 人类与专家产生分歧 (Divergence)", fg=DIFF_COLOR)

        # 态势评估正名（优势/均势/劣势，而非绝对终局）
        e_score = item.get("expert_score", 0.0)
        if e_score >= 300.0:
            adv_str = "优势局面 (明子/控制明显占优)"
            adv_col = "#27AE60"
        elif e_score <= -300.0:
            adv_str = "劣势局面 (明显落后/受制)"
            adv_col = "#C0392B"
        else:
            adv_str = "均势拉锯 (相持互角态势)"
            adv_col = "#2980B9"

        self.lbl_pos_adv.config(text=f"盘面态势参考:  {adv_str}", fg=adv_col)
        d_info = f" (Depth {e_depth} 超深推演)" if e_depth >= 5 else " (Depth 3 + QSearch)"
        self.lbl_expert_score.config(text=f"专家战术评分:  {e_score:+.1f}{d_info}")

        # 5 层超深推演状态展示与按钮更新
        if e_depth >= 5:
            d5_t = item.get("expert_d5_time", 0.0)
            self.lbl_depth5_status.config(
                text=f"✅ 该局面已完成 5 层超深推演 (耗时 {d5_t}s)", fg="#27AE60"
            )
            self.btn_depth5.config(text="⚡ 重新进行 5 层推演 (Depth 5)")
        else:
            self.lbl_depth5_status.config(
                text="当前为 3 层初筛标签；遇到疑难局可点击上方按钮超算", fg="#888"
            )
            self.btn_depth5.config(text="⚡ 启动 5 层超深度推演 (Depth 5)")

        # 4. 当前裁决状态显示
        verified = item.get("human_verified", False)
        judgment = item.get("action_judgment")
        if verified and judgment in JUDGMENTS:
            name, fg_col, _ = JUDGMENTS[judgment]
            self.lbl_current_label.config(text=f"【人工已裁决】: {name}", fg=fg_col)
        elif verified and item.get("verified_label") is not None:
            v_code = item.get("verified_label")
            legacy_names = {0: "⭐ 专家更优", 1: "🤝 两手皆可", 2: "❌ 专家恶手", 3: "👤 人类更佳", -1: "❓ 存疑"}
            self.lbl_current_label.config(text=f"【人工已核验】: {legacy_names.get(v_code, '已核验')}", fg="#27AE60")
        else:
            self.lbl_current_label.config(text="【待裁决】: 尚未人工审定", fg="#777")

        self.comment_var.set(item.get("comment", ""))

        # 5. 绘制棋盘与高亮
        self.draw_board()
        self.draw_pieces()
        self.draw_action_overlays()
        self.update_stats()

    def format_action(self, st: GameState, act: Optional[Action]) -> str:
        if act is None:
            return "无走法"
        if act.kind == "flip":
            pc = st.board.get(act.frm)
            name = piece_label(pc) if pc else "暗子"
            return f"翻({act.frm[0]},{act.frm[1]}) [{name}]"
        mover = st.board.get(act.frm)
        mover_name = piece_label(mover) if mover else "棋子"
        target = st.board.get(act.to)
        tgt_name = piece_label(target) if target else "空地"
        return f"移({act.frm[0]},{act.frm[1]})->({act.to[0]},{act.to[1]}) [{mover_name} 击 {tgt_name}]"

    # ------------------------------------------------------------- 棋盘绘制

    def draw_board(self):
        cv = self.canvas
        cv.delete("all")
        cv.create_rectangle(6, TITLE_H, BOARD_W - 6, BOARD_H - 6,
                            fill=WOOD, outline=WOOD_DARK, width=4)
        cv.create_text(BOARD_W / 2, TITLE_H / 2 + 4, text="军  棋  核  验  面  板",
                       font=FONT_TITLE, fill="#2F5D1E")

        # 公路
        for p, ns in NEIGHBORS.items():
            for q in ns:
                if p < q:
                    x0, y0 = rc_to_xy(*p)
                    x1, y1 = rc_to_xy(*q)
                    cv.create_line(x0, y0, x1, y1, fill=ROAD_FILL, width=3)
        # 铁路
        for p in ALL_POSITIONS:
            if not is_rail(p):
                continue
            for q in rail_neighbors(p):
                if p < q:
                    x0, y0 = rc_to_xy(*p)
                    x1, y1 = rc_to_xy(*q)
                    cv.create_line(x0, y0, x1, y1, fill=RAIL_FILL, width=5, dash=(10, 6))

        for (r, c) in CAMPS:
            x, y = rc_to_xy(r, c)
            cv.create_oval(x - 24, y - 16, x + 24, y + 16, outline="#4C7A2E", width=4)

        for (r, c) in HQS:
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
                cv.create_rectangle(x0, y0, x1, y1, fill=face_color(pc), outline="#222", width=1)
                cv.create_text(x, y, text=RANK_CN[pc.rank], font=FONT_PIECE, fill=FACE_TEXT)
            else:
                x0, y0 = x - CELL_W / 2 + 4, y - CELL_H / 2 + 3
                x1, y1 = x + CELL_W / 2 - 4, y + CELL_H / 2 - 3
                cv.create_rectangle(x0, y0, x1, y1, fill=BACK_FILL, outline=BACK_EDGE, width=2)
                cv.create_rectangle(x0 + 5, y0 + 4, x1 - 5, y1 - 4, outline="#79A855", width=1)

    def draw_action_overlays(self):
        """高亮人类走法 (青蓝) 与专家推荐走法 (橙金)。"""
        cv = self.canvas

        def _draw_act(act: Optional[Action], color: str, offset: int, tag_text: str):
            if act is None:
                return
            fx, fy = rc_to_xy(*act.frm)
            # 起点框
            cv.create_rectangle(fx - CELL_W / 2 + offset, fy - CELL_H / 2 + offset,
                                fx + CELL_W / 2 - offset, fy + CELL_H / 2 - offset,
                                outline=color, width=3)
            cv.create_text(fx, fy - CELL_H / 2 + 8, text=tag_text, font=FONT_S, fill=color)

            if act.kind == "move" and act.to is not None:
                tx, ty = rc_to_xy(*act.to)
                cv.create_rectangle(tx - CELL_W / 2 + offset, ty - CELL_H / 2 + offset,
                                    tx + CELL_W / 2 - offset, ty + CELL_H / 2 - offset,
                                    outline=color, width=2, dash=(6, 3))
                # 绘制移动指向箭头
                cv.create_line(fx, fy, tx, ty, fill=color, width=3, arrow="last", arrowshape=(12, 14, 5))

        # 1. 人类实战动作 (青蓝)
        _draw_act(self.human_action, HUMAN_MOVE_COLOR, 1, "人")
        # 2. 专家推荐动作 (橙金)
        _draw_act(self.expert_action, EXPERT_MOVE_COLOR, 3, "专")

    # ------------------------------------------------------------- 导航与打标

    def next_item(self):
        if not self.filtered_indices:
            return
        if self.current_idx_in_filter < len(self.filtered_indices) - 1:
            self.current_idx_in_filter += 1
            self.show_current()
        else:
            messagebox.showinfo("提示", "已到达当前列表最后一局！")

    def prev_item(self):
        if not self.filtered_indices:
            return
        if self.current_idx_in_filter > 0:
            self.current_idx_in_filter -= 1
            self.show_current()
        else:
            messagebox.showinfo("提示", "已是当前列表第一局！")

    def random_item(self):
        if not self.filtered_indices:
            return
        self.current_idx_in_filter = random.randrange(len(self.filtered_indices))
        self.show_current()

    def jump_to_id(self):
        text = self.jump_var.get().strip()
        if not text.isdigit():
            return
        target_id = int(text)
        # 在 filtered_indices 中查找目标 ID
        for f_idx, act_idx in enumerate(self.filtered_indices):
            if self.dataset[act_idx].get("id") == target_id:
                self.current_idx_in_filter = f_idx
                self.show_current()
                return
        messagebox.showinfo("未找到", f"在当前过滤列表中未找到 ID #{target_id}")

    def set_judgment(self, judgment: str):
        """人工招法合理性裁决并自动切至下一局。"""
        item = self.get_current_item()
        if item is None:
            return
        item["human_verified"] = True
        item["action_judgment"] = judgment
        # 兼容旧编码字段
        legacy_map = {"expert_better": 0, "both_fine": 1, "expert_blunder": 2, "human_better": 3, "skip": -1}
        item["verified_label"] = legacy_map.get(judgment, -1)
        item["comment"] = self.comment_var.get().strip()

        # 自动切换到下一个
        if self.current_idx_in_filter < len(self.filtered_indices) - 1:
            self.current_idx_in_filter += 1
        self.show_current()

    def set_label(self, label: int):
        """兼容性包装：旧版数字判定重定向至 set_judgment。"""
        reverse_map = {0: "expert_better", 1: "both_fine", 2: "expert_blunder", 3: "human_better", -1: "skip"}
        self.set_judgment(reverse_map.get(label, "skip"))

    def start_depth5_search(self):
        """异步启动当前局面的 5 层超深度专家推演 (选项 B)。"""
        if self.is_searching:
            return
        item = self.get_current_item()
        if item is None or getattr(self, "state", None) is None:
            return

        self.is_searching = True
        self.btn_depth5.config(state="disabled", text="⏳ 5 层超深推演中 (约需 10~30s)...")
        self.lbl_depth5_status.config(text="正在进行 5 层极大极小 + 4 层 QSearch 深度推演...", fg="#E67E22")

        st = self.state.copy()
        item_ref = item

        def _worker():
            from .ai import ExpertAgent
            from .config import SearchConfig
            t0 = time.perf_counter()
            try:
                # 60秒超时保护
                agent = ExpertAgent(SearchConfig(depth=5, time_limit_ms=60_000), seed=42)
                best_act, best_score, stats = agent.engine.search(st, max_depth=5, time_limit_ms=60_000)
                elapsed = time.perf_counter() - t0
                self.root.after(0, lambda: self._on_depth5_done(item_ref, best_act, best_score, stats, elapsed))
            except Exception as ex:
                self.root.after(0, lambda: self._on_depth5_error(str(ex)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_depth5_done(self, item, best_act, best_score, stats, elapsed):
        self.is_searching = False
        self.btn_depth5.config(state="normal", text="⚡ 重新进行 5 层推演 (Depth 5)")
        if best_act is not None:
            item["expert_action_d5"] = action_to_dict(best_act)
            item["expert_score_d5"] = float(best_score)
            item["expert_depth"] = 5
            item["expert_d5_time"] = round(elapsed, 2)
            item["expert_d5_nodes"] = stats.nodes
            item["expert_d5_qnodes"] = stats.qnodes

            # 同步替换主推荐动作并重新比对
            item["expert_action"] = action_to_dict(best_act)
            item["expert_score"] = float(best_score)
            h_act = dict_to_action(item.get("human_action"))
            item["human_expert_match"] = (best_act == h_act)

            self.lbl_depth5_status.config(
                text=f"✅ 5 层推演完成: 耗时 {elapsed:.1f}s | 展开 {stats.nodes} (主) + {stats.qnodes} (战术)",
                fg="#27AE60"
            )
            # 刷新棋盘和走法高亮
            self.show_current()
        else:
            self.lbl_depth5_status.config(text=f"⚠️ 未能生成有效走法 (耗时 {elapsed:.1f}s)", fg="#C0392B")

    def _on_depth5_error(self, err_msg):
        self.is_searching = False
        self.btn_depth5.config(state="normal", text="⚡ 启动 5 层超深度推演 (Depth 5)")
        self.lbl_depth5_status.config(text=f"❌ 5 层推演异常: {err_msg}", fg="#C0392B")

    def update_comment(self):
        item = self.get_current_item()
        if item is not None:
            item["comment"] = self.comment_var.get().strip()

    def update_stats(self):
        if not self.dataset:
            self.lbl_stats.config(text="无数据")
            return
        total = len(self.dataset)
        verified_count = sum(1 for p in self.dataset if p.get("human_verified"))
        counts = Counter(p.get("action_judgment") for p in self.dataset if p.get("human_verified"))
        d5_count = sum(1 for p in self.dataset if p.get("expert_depth") == 5)

        self.lbl_stats.config(
            text=f"总样本: {total} | 已核验: {verified_count} ({verified_count/total*100:.1f}%) | 5层超算: {d5_count}\n"
                 f"⭐专家优: {counts.get('expert_better', 0)} | 👤人类佳: {counts.get('human_better', 0)} | "
                 f"🤝皆可: {counts.get('both_fine', 0)} | ❌恶手: {counts.get('expert_blunder', 0)}"
        )

    def save_data(self):
        """保存已核验数据集至磁盘。"""
        if not self.dataset:
            return
        try:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self.dataset, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("保存成功", f"核验数据集已成功保存至:\n{self.data_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存出错: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="军棋实战交火打标与核验 GUI 工作台")
    parser.add_argument("--data", default="datasets/distill_tactical_labeled.json",
                        help="待核验打标数据集 JSON 路径")
    args = parser.parse_args(argv)

    root = tk.Tk()
    app = LabelInspectorApp(root, data_path=args.data)
    root.mainloop()


if __name__ == "__main__":
    main()

