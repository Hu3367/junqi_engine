"""军棋 AI 实验靶场与多维指标看板模块 (Junqi AI Benchmark and Metrics Suite)。

学术依据与设计原则：
1. 建立固定可解残局与中盘题库（Ground Truth Tablebase/Expert Suite），计算 Value MAE 与 Policy 吻合度；
2. 自动化追踪：胜率 (Win Rate)、和棋率 (Draw Rate)、重复率 (Repetition Rate)、平均手数 (Avg Plies)、Elo 梯队；
3. 输出结构化 JSON 指标至 metrics/ 目录，支持 Champion vs Challenger 严格门控评估。
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .ai import (Agent, ApkNativeAgent, ExpertSearchEngine, HybridAgent, NNAgent,
                evaluate_expert)
from .config import EvalWeights, RuleConfig, SearchConfig
from .encoder import action_to_index, encode_state, encode_state_np, legal_action_mask
from .net import JunqiNet
from .rules import CAMPS, HQS, PLAY_POSITIONS, Rank, is_camp, is_hq, is_rail, other
from .state import Action, GameState, Piece, deal, position_key


# ---------------------------------------------------------------- 标杆测试用例构造

def create_benchmark_suite() -> list[dict]:
    """构造包含开局 (15)、中盘攻防 (15)、死区与定式残局 (20) 的 50 个固定标杆局面（附带专家真值）。"""
    cfg = RuleConfig()
    suite = []

    # =========================================================================
    # 1. 经典尾盘残局定式题库 (20 题)
    # =========================================================================

    # 1.1 红司令+工兵 vs 蓝单地雷+军旗 (红必胜)
    st_eg01 = GameState(board={
        (10, 0): Piece("r", Rank.GONG, True),
        (8, 2): Piece("r", Rank.SI, True),
        (0, 1): Piece("b", Rank.QI, True),
        (1, 1): Piece("b", Rank.LEI, True),
    }, turn=0, cfg=cfg)
    st_eg01.seat_color[0], st_eg01.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_01_win_flag_mine", "category": "endgame", "state": st_eg01, "true_value": 1.0, "true_val_class": 0, "key_action": Action("move", (10, 0), (9, 0))})

    # 1.2 双死区铁壁残局 (必和)
    st_eg02 = GameState(board={
        (11, 1): Piece("r", Rank.QI, True), (10, 1): Piece("r", Rank.LEI, True), (5, 2): Piece("r", Rank.TUAN, True),
        (0, 1): Piece("b", Rank.QI, True), (1, 1): Piece("b", Rank.LEI, True), (6, 2): Piece("b", Rank.TUAN, True),
    }, turn=0, cfg=cfg)
    st_eg02.seat_color[0], st_eg02.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_02_draw_fortress", "category": "endgame", "state": st_eg02, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 1.3 红单排长 vs 蓝军长+师长 (红必败)
    st_eg03 = GameState(board={
        (5, 2): Piece("r", Rank.PAI, True), (11, 1): Piece("r", Rank.QI, True),
        (4, 2): Piece("b", Rank.JUN, True), (6, 2): Piece("b", Rank.SHI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg03.seat_color[0], st_eg03.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_03_loss_outnumbered", "category": "endgame", "state": st_eg03, "true_value": -1.0, "true_val_class": 2, "key_action": None})

    # 1.4 红单工兵 vs 蓝单雷一旗 (红必胜)
    st_eg04 = GameState(board={
        (6, 0): Piece("r", Rank.GONG, True), (11, 1): Piece("r", Rank.QI, True),
        (1, 1): Piece("b", Rank.LEI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg04.seat_color[0], st_eg04.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_04_gong_vs_flag_mine", "category": "endgame", "state": st_eg04, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # 1.5 均势单排长对峙 (必和)
    st_eg05 = GameState(board={
        (8, 2): Piece("r", Rank.PAI, True), (11, 1): Piece("r", Rank.QI, True),
        (3, 2): Piece("b", Rank.PAI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg05.seat_color[0], st_eg05.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_05_pai_vs_pai", "category": "endgame", "state": st_eg05, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 1.6 均势单团长对峙 (必和)
    st_eg06 = GameState(board={
        (6, 0): Piece("r", Rank.TUAN, True), (11, 1): Piece("r", Rank.QI, True),
        (5, 0): Piece("b", Rank.TUAN, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg06.seat_color[0], st_eg06.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_06_tuan_vs_tuan", "category": "endgame", "state": st_eg06, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 1.7 连长碰营长 (劣势必败)
    st_eg07 = GameState(board={
        (5, 2): Piece("r", Rank.LIAN, True), (11, 1): Piece("r", Rank.QI, True),
        (4, 2): Piece("b", Rank.YING, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg07.seat_color[0], st_eg07.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_07_lian_vs_ying", "category": "endgame", "state": st_eg07, "true_value": -1.0, "true_val_class": 2, "key_action": None})

    # 1.8 司令碾压军长 (红必胜)
    st_eg08 = GameState(board={
        (5, 2): Piece("r", Rank.SI, True), (11, 1): Piece("r", Rank.QI, True),
        (4, 2): Piece("b", Rank.JUN, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg08.seat_color[0], st_eg08.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_08_si_vs_jun", "category": "endgame", "state": st_eg08, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # 1.9 师长碰司令 (红必败)
    st_eg09 = GameState(board={
        (5, 2): Piece("r", Rank.SHI, True), (11, 1): Piece("r", Rank.QI, True),
        (4, 2): Piece("b", Rank.SI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg09.seat_color[0], st_eg09.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_09_shi_vs_si", "category": "endgame", "state": st_eg09, "true_value": -1.0, "true_val_class": 2, "key_action": None})

    # 1.10 蓝方无合法走法困毙 (红必胜)
    st_eg10 = GameState(board={
        (6, 2): Piece("r", Rank.SI, True), (11, 1): Piece("r", Rank.QI, True),
        (0, 1): Piece("b", Rank.QI, True), (1, 1): Piece("b", Rank.LEI, True),
    }, turn=1, cfg=cfg)
    st_eg10.seat_color[0], st_eg10.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_10_immobilized_win", "category": "endgame", "state": st_eg10, "true_value": -1.0, "true_val_class": 2, "key_action": None})

    # 1.11 双方各自锁死在地雷后方 (必和)
    st_eg11 = GameState(board={
        (11, 1): Piece("r", Rank.QI, True), (10, 1): Piece("r", Rank.LEI, True), (10, 3): Piece("r", Rank.LEI, True), (11, 2): Piece("r", Rank.PAI, True),
        (0, 1): Piece("b", Rank.QI, True), (1, 1): Piece("b", Rank.LEI, True), (1, 3): Piece("b", Rank.LEI, True), (0, 2): Piece("b", Rank.PAI, True),
    }, turn=0, cfg=cfg)
    st_eg11.seat_color[0], st_eg11.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_11_mine_wall_draw", "category": "endgame", "state": st_eg11, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 1.12 工兵一步冲刺吃旗 (红必胜)
    st_eg12 = GameState(board={
        (1, 0): Piece("r", Rank.GONG, True), (11, 1): Piece("r", Rank.QI, True),
        (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg12.seat_color[0], st_eg12.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_12_gong_speed_flag", "category": "endgame", "state": st_eg12, "true_value": 1.0, "true_val_class": 0, "key_action": Action("move", (1, 0), (0, 1))})

    # 1.13 双工兵扫清地雷夺旗 (红必胜)
    st_eg13 = GameState(board={
        (6, 0): Piece("r", Rank.GONG, True), (6, 4): Piece("r", Rank.GONG, True), (11, 1): Piece("r", Rank.QI, True),
        (1, 1): Piece("b", Rank.LEI, True), (1, 3): Piece("b", Rank.LEI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg13.seat_color[0], st_eg13.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_13_double_gong_vs_mine", "category": "endgame", "state": st_eg13, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # 1.14 红单司令（无工兵）遇地雷护旗 (必和)
    st_eg14 = GameState(board={
        (5, 2): Piece("r", Rank.SI, True), (11, 1): Piece("r", Rank.QI, True),
        (1, 1): Piece("b", Rank.LEI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg14.seat_color[0], st_eg14.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_14_single_si_no_gong_draw", "category": "endgame", "state": st_eg14, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 1.15 行营避难持久战 (必和)
    st_eg15 = GameState(board={
        (7, 1): Piece("r", Rank.PAI, True), (11, 1): Piece("r", Rank.QI, True),
        (6, 1): Piece("b", Rank.SI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg15.seat_color[0], st_eg15.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_15_camp_stalemate", "category": "endgame", "state": st_eg15, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 1.16 红军长+工兵 vs 蓝单雷一旗 (红必胜)
    st_eg16 = GameState(board={
        (5, 2): Piece("r", Rank.JUN, True), (5, 0): Piece("r", Rank.GONG, True), (11, 1): Piece("r", Rank.QI, True),
        (1, 1): Piece("b", Rank.LEI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg16.seat_color[0], st_eg16.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_16_jun_gong_win", "category": "endgame", "state": st_eg16, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # 1.17 双方均无工兵且有地雷 (必和)
    st_eg17 = GameState(board={
        (11, 1): Piece("r", Rank.QI, True), (10, 1): Piece("r", Rank.LEI, True), (8, 2): Piece("r", Rank.SI, True),
        (0, 1): Piece("b", Rank.QI, True), (1, 1): Piece("b", Rank.LEI, True), (3, 2): Piece("b", Rank.SI, True),
    }, turn=0, cfg=cfg)
    st_eg17.seat_color[0], st_eg17.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_17_no_gong_left_draw", "category": "endgame", "state": st_eg17, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 1.18 蓝方大子成群控场 (红必败)
    st_eg18 = GameState(board={
        (10, 2): Piece("r", Rank.PAI, True), (11, 1): Piece("r", Rank.QI, True),
        (5, 2): Piece("b", Rank.SI, True), (6, 2): Piece("b", Rank.JUN, True), (7, 2): Piece("b", Rank.SHI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg18.seat_color[0], st_eg18.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_18_overwhelming_loss", "category": "endgame", "state": st_eg18, "true_value": -1.0, "true_val_class": 2, "key_action": None})

    # 1.19 炸弹同归于尽残局 (必和)
    st_eg19 = GameState(board={
        (5, 2): Piece("r", Rank.ZHA, True), (11, 1): Piece("r", Rank.QI, True),
        (4, 2): Piece("b", Rank.SI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg19.seat_color[0], st_eg19.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_19_zha_clash_draw", "category": "endgame", "state": st_eg19, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 1.20 双师长围攻单营长 (红必胜)
    st_eg20 = GameState(board={
        (5, 1): Piece("r", Rank.SHI, True), (5, 3): Piece("r", Rank.SHI, True), (11, 1): Piece("r", Rank.QI, True),
        (4, 2): Piece("b", Rank.YING, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_eg20.seat_color[0], st_eg20.seat_color[1] = "r", "b"
    suite.append({"id": "endgame_20_double_shi_win", "category": "endgame", "state": st_eg20, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # =========================================================================
    # 2. 中盘战术与攻防题库 (15 题)
    # S0 修复：true_value 全部取类别一致极端值（Win→+1 / Loss→−1 / Draw→0），
    # 消除连续真值与三分类头数学互斥导致的不可达 MAE 目标。
    # =========================================================================

    # 2.1 司令行营避险
    st_mg01 = GameState(board={
        (3, 2): Piece("r", Rank.SI, True), (4, 2): Piece("b", Rank.ZHA, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg01.seat_color[0], st_mg01.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_01_camp_dodge", "category": "midgame", "state": st_mg01, "true_value": 1.0, "true_val_class": 0, "key_action": Action("move", (3, 2), (2, 2))})

    # 2.2 工兵破路挖雷
    st_mg02 = GameState(board={
        (6, 0): Piece("r", Rank.GONG, True), (5, 0): Piece("b", Rank.LEI, True), (5, 2): Piece("r", Rank.JUN, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg02.seat_color[0], st_mg02.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_02_gong_mine_clear", "category": "midgame", "state": st_mg02, "true_value": 1.0, "true_val_class": 0, "key_action": Action("move", (6, 0), (5, 0))})

    # 2.3 炸弹换司令
    st_mg03 = GameState(board={
        (5, 2): Piece("r", Rank.ZHA, True), (4, 2): Piece("b", Rank.SI, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg03.seat_color[0], st_mg03.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_03_zha_trade_si", "category": "midgame", "state": st_mg03, "true_value": 1.0, "true_val_class": 0, "key_action": Action("move", (5, 2), (4, 2))})

    # 2.4 占领中央行营控制局势
    st_mg04 = GameState(board={
        (7, 2): Piece("r", Rank.JUN, True), (3, 2): Piece("b", Rank.SHI, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg04.seat_color[0], st_mg04.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_04_camp_control", "category": "midgame", "state": st_mg04, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # 2.5 红主力被围剿陷阱 (中盘劣势)
    st_mg05 = GameState(board={
        (5, 2): Piece("r", Rank.SHI, True),
        (4, 2): Piece("b", Rank.SI, True), (6, 2): Piece("b", Rank.JUN, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg05.seat_color[0], st_mg05.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_05_pinned_loss", "category": "midgame", "state": st_mg05, "true_value": -1.0, "true_val_class": 2, "key_action": None})

    # 2.6 中盘均势胶着
    st_mg06 = GameState(board={
        (7, 1): Piece("r", Rank.SI, True), (7, 3): Piece("r", Rank.JUN, True),
        (4, 1): Piece("b", Rank.SI, True), (4, 3): Piece("b", Rank.JUN, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg06.seat_color[0], st_mg06.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_06_equal_balance", "category": "midgame", "state": st_mg06, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 2.7 侧翼工兵包抄
    st_mg07 = GameState(board={
        (5, 0): Piece("r", Rank.GONG, True), (6, 2): Piece("r", Rank.SI, True),
        (3, 4): Piece("b", Rank.SHI, True), (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg07.seat_color[0], st_mg07.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_07_flank_flier", "category": "midgame", "state": st_mg07, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # 2.8 敌司令逼近大本营 (红危急)
    st_mg08 = GameState(board={
        (10, 1): Piece("b", Rank.SI, True), (11, 1): Piece("r", Rank.QI, True), (11, 2): Piece("r", Rank.PAI, True),
        (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg08.seat_color[0], st_mg08.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_08_hq_under_fire", "category": "midgame", "state": st_mg08, "true_value": -1.0, "true_val_class": 2, "key_action": None})

    # 2.9 炸弹伏击中路
    st_mg09 = GameState(board={
        (6, 2): Piece("r", Rank.ZHA, True), (4, 2): Piece("b", Rank.JUN, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg09.seat_color[0], st_mg09.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_09_zha_ambush", "category": "midgame", "state": st_mg09, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # 2.10 铁路阻击
    st_mg10 = GameState(board={
        (6, 4): Piece("r", Rank.TUAN, True), (5, 4): Piece("b", Rank.GONG, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg10.seat_color[0], st_mg10.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_10_rail_block", "category": "midgame", "state": st_mg10, "true_value": 1.0, "true_val_class": 0, "key_action": Action("move", (6, 4), (5, 4))})

    # 2.11 双方铁壁防御中盘
    st_mg11 = GameState(board={
        (8, 2): Piece("r", Rank.SI, True), (9, 1): Piece("r", Rank.LEI, True), (11, 1): Piece("r", Rank.QI, True),
        (3, 2): Piece("b", Rank.SI, True), (2, 1): Piece("b", Rank.LEI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg11.seat_color[0], st_mg11.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_11_fortress_mid", "category": "midgame", "state": st_mg11, "true_value": 0.0, "true_val_class": 1, "key_action": None})

    # 2.12 司令前线清场
    st_mg12 = GameState(board={
        (5, 2): Piece("r", Rank.SI, True), (5, 1): Piece("b", Rank.PAI, True), (5, 3): Piece("b", Rank.LIAN, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg12.seat_color[0], st_mg12.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_12_si_sweep", "category": "midgame", "state": st_mg12, "true_value": 1.0, "true_val_class": 0, "key_action": Action("move", (5, 2), (5, 1))})

    # 2.13 双线受敌 (红劣势)
    st_mg13 = GameState(board={
        (7, 2): Piece("r", Rank.LV, True), (6, 1): Piece("b", Rank.SI, True), (8, 3): Piece("b", Rank.JUN, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg13.seat_color[0], st_mg13.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_13_two_front_loss", "category": "midgame", "state": st_mg13, "true_value": -1.0, "true_val_class": 2, "key_action": None})

    # 2.14 敌军旗暴露工兵突击
    st_mg14 = GameState(board={
        (2, 0): Piece("r", Rank.GONG, True), (1, 1): Piece("b", Rank.PAI, True), (0, 1): Piece("b", Rank.QI, True),
        (11, 1): Piece("r", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg14.seat_color[0], st_mg14.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_14_flag_exposed_rush", "category": "midgame", "state": st_mg14, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # 2.15 诱敌牺牲战术
    st_mg15 = GameState(board={
        (5, 2): Piece("r", Rank.LIAN, True), (6, 2): Piece("r", Rank.SI, True), (4, 2): Piece("b", Rank.LV, True),
        (11, 1): Piece("r", Rank.QI, True), (0, 1): Piece("b", Rank.QI, True),
    }, turn=0, cfg=cfg)
    st_mg15.seat_color[0], st_mg15.seat_color[1] = "r", "b"
    suite.append({"id": "midgame_15_tactical_bait", "category": "midgame", "state": st_mg15, "true_value": 1.0, "true_val_class": 0, "key_action": None})

    # =========================================================================
    # 3. 开局标准发牌题库 (15 题)
    # =========================================================================
    for seed in range(2026, 2041):
        st_open = deal(random.Random(seed), cfg)
        suite.append({
            "id": f"opening_seed_{seed}",
            "category": "opening",
            "state": st_open,
            "true_value": 0.0,
            "true_val_class": 1,
            "key_action": None,
        })

    return suite


# ---------------------------------------------------------------- 评估指标计算

def evaluate_net_benchmark(net: JunqiNet, suite: Optional[list[dict]] = None,
                           device: str = "cpu") -> dict:
    """在标杆题库上评估神经网络的 Value MAE、三分类准确率与 Policy 表现（含类别分布监控）。"""
    suite = suite or create_benchmark_suite()
    net.eval()

    val_errors = []
    class_correct = 0
    class_total = 0
    cat_mae = {"endgame": [], "midgame": [], "opening": []}
    pred_counts = {0: 0, 1: 0, 2: 0}  # 0=Win, 1=Draw, 2=Loss
    true_counts = {0: 0, 1: 0, 2: 0}
    brier_sum = 0.0
    entropy_sum = 0.0

    for item in suite:
        st = item["state"]
        true_val = item.get("true_value")
        true_cls = item.get("true_val_class")
        cat = item.get("category", "midgame")

        policy_map, scalar_val = net.predict_state(st, seat=st.turn, device=device)
        _, val_probs = net.predict_probabilities(st, seat=st.turn, device=device)

        if true_val is not None:
            err = abs(scalar_val - true_val)
            val_errors.append(err)
            if cat in cat_mae:
                cat_mae[cat].append(err)

        if true_cls is not None:
            class_total += 1
            true_counts[true_cls] = true_counts.get(true_cls, 0) + 1
            probs_vec = (val_probs["win"], val_probs["draw"], val_probs["loss"])
            pred_cls = int(np.argmax(probs_vec))
            pred_counts[pred_cls] = pred_counts.get(pred_cls, 0) + 1
            if pred_cls == true_cls:
                class_correct += 1
            # Brier 分数与预测熵（S0 新增：类别概率校准监控）
            brier_sum += sum((probs_vec[i] - (1.0 if i == true_cls else 0.0)) ** 2
                             for i in range(3))
            entropy_sum += -sum(pv * math.log(max(pv, 1e-9)) for pv in probs_vec)

    overall_mae = float(np.mean(val_errors)) if val_errors else 0.0
    val_acc = (class_correct / class_total) if class_total else 0.0
    brier = (brier_sum / class_total) if class_total else 0.0
    mean_entropy = (entropy_sum / class_total) if class_total else 0.0

    # 塌缩监控（S0 收紧：单一类别预测占比 >= 70% 即判定为塌缩）
    max_pred_prop = (max(pred_counts.values()) / class_total) if class_total else 0.0
    collapse_warning = bool(max_pred_prop >= 0.70)

    return {
        "benchmark_total": len(suite),
        "value_mae_overall": overall_mae,
        "value_class_acc": val_acc,
        "value_mae_endgame": float(np.mean(cat_mae["endgame"])) if cat_mae["endgame"] else 0.0,
        "value_mae_midgame": float(np.mean(cat_mae["midgame"])) if cat_mae["midgame"] else 0.0,
        "value_mae_opening": float(np.mean(cat_mae["opening"])) if cat_mae["opening"] else 0.0,
        "pred_class_counts": {"Win": pred_counts[0], "Draw": pred_counts[1], "Loss": pred_counts[2]},
        "true_class_counts": {"Win": true_counts[0], "Draw": true_counts[1], "Loss": true_counts[2]},
        "value_brier": brier,
        "value_pred_entropy": mean_entropy,
        "collapse_warning": collapse_warning,
    }


# ---------------------------------------------------------------- 对抗评测与 Elo 梯队

def run_tournament_match(agent_a: Agent, agent_b: Agent, n_games: int = 40,
                         base_seed: int = 2026, cfg: Optional[RuleConfig] = None) -> dict:
    """双雄争霸赛：进行镜像轮换对局，计算胜率、和棋率、重复率与 Elo 差异。"""
    cfg = cfg or RuleConfig()
    wins_a = 0
    wins_b = 0
    draws = 0
    repetition_draws = 0
    total_plies = 0

    for i in range(n_games):
        seed = base_seed + i
        # 偶数局 A 执先(0)，奇数局 B 执先(0)
        a_seat = 0 if (i % 2 == 0) else 1
        b_seat = 1 - a_seat

        agents = {a_seat: agent_a, b_seat: agent_b}
        st = deal(random.Random(seed), cfg)
        pos_seen = {position_key(st): 1}

        while not st.is_terminal():
            mover = agents[st.turn]
            act = mover.select_action(st)
            st = st.apply(act)
            total_plies += 1
            pk = position_key(st)
            pos_seen[pk] = pos_seen.get(pk, 0) + 1

        if st.winner == -1:
            draws += 1
            if st.win_reason == "repetition":
                repetition_draws += 1
        elif st.winner == a_seat:
            wins_a += 1
        elif st.winner == b_seat:
            wins_b += 1

    score_rate_a = (wins_a + 0.5 * draws) / n_games if n_games else 0.5
    draw_rate = draws / n_games if n_games else 0.0
    rep_rate = repetition_draws / n_games if n_games else 0.0
    avg_len = total_plies / n_games if n_games else 0.0

    # Elo 差值计算: delta = -400 * log10(1/score - 1)
    clipped_score = max(0.01, min(0.99, score_rate_a))
    elo_diff = -400.0 * math.log10(1.0 / clipped_score - 1.0)

    return {
        "games": n_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "repetition_draws": repetition_draws,
        "score_rate_a": score_rate_a,
        "win_rate_a": wins_a / n_games if n_games else 0.0,
        "draw_rate": draw_rate,
        "repetition_rate": rep_rate,
        "avg_plies": avg_len,
        "elo_diff": elo_diff,
    }


def run_apk_challenge(candidate_agent, n_games: int = 40, level: str = "advanced",
                      base_seed: int = 2026, cfg: Optional[RuleConfig] = None) -> dict:
    """P4 假想敌挑战赛：将自研候选模型与官方 APK 原版智能体 (ApkNativeAgent) 进行对抗评测。

    参数:
        candidate_agent: 参评自研智能体 (Agent/ExpertAgent/HybridAgent/NNAgent)
        n_games: 对局局数 (先后手各半)
        level: APK 原生假想敌难度 ("beginner", "intermediate", "advanced")
        base_seed: 基础发牌随机种子
        cfg: 规则开关
    返回:
        字典包含对抗得分率、胜率、和棋率、重复率、平均手数及相对 Elo 分差。
    """
    opponent = ApkNativeAgent(level=level, seed=base_seed)
    res = run_tournament_match(candidate_agent, opponent, n_games=n_games,
                               base_seed=base_seed, cfg=cfg)
    res["apk_level"] = level
    return res


# ---------------------------------------------------------------- 看板日志持久化

def save_metrics_report(metrics: dict, output_dir: str = "metrics"):
    """将指标分别持久化到 metrics/ 目录下的独立 JSON 与 Markdown 仪表盘。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 独立 JSON 指标文件
    for key, val in metrics.items():
        fname = os.path.join(output_dir, f"{key}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(val, f, indent=2, ensure_ascii=False)

    # 2. 生成可视化 Markdown 看板
    md_path = os.path.join(output_dir, "benchmark_dashboard.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 军棋 AI 实验靶场多维指标看板 (Benchmark Dashboard)\n\n")
        f.write(f"> **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 核心指标一览\n\n")
        f.write("| 指标项 | 当前数值 | 评估目标 |\n")
        f.write("|---|---|---|\n")
        if "value_mae" in metrics:
            v_info = metrics["value_mae"]
            mae = v_info.get("value_mae_overall", 0.0)
            acc = v_info.get("value_class_acc", 0.0) * 100.0
            tot = v_info.get("benchmark_total", 0)
            preds = v_info.get("pred_class_counts", {})
            trues = v_info.get("true_class_counts", {})
            collapse = v_info.get("collapse_warning", False)
            f.write(f"| **标杆题库覆盖量 (Total Suite)** | `{tot} 题` | 固定 50 题标杆集 |\n")
            f.write(f"| **Value MAE (总体绝对误差)** | `{mae:.4f}` | 持续下降 (< 0.1500，S0 后真值与类别一致) |\n")
            f.write(f"| **Value 三分类准确率** | `{acc:.1f}%` | > 85.0% |\n")
            f.write(f"| **Brier 分数 (类别概率校准)** | `{v_info.get('value_brier', 0.0):.4f}` | < 0.5000 |\n")
            f.write(f"| **预测熵 (最大 {math.log(3.0):.4f})** | `{v_info.get('value_pred_entropy', 0.0):.4f}` | 监控塌缩趋势 |\n")
            f.write(f"| **预测类别分布 (Win/Draw/Loss)** | `胜 {preds.get('Win',0)} / 和 {preds.get('Draw',0)} / 负 {preds.get('Loss',0)}` (真值: `胜 {trues.get('Win',0)} / 和 {trues.get('Draw',0)} / 负 {trues.get('Loss',0)}`) | 避免单一类别塌缩 |\n")
            if collapse:
                f.write(f"| **类别塌缩报警 (Collapse Alert)** | `🚨 严重警告：单一类别预测占比超过 70%` | 必须破除塌缩闭环 |\n")
            f.write(f"| **开局误差 (Opening MAE)** | `{v_info.get('value_mae_opening', 0.0):.4f}` | < 0.0500 |\n")
            f.write(f"| **中盘攻防误差 (Midgame MAE)** | `{v_info.get('value_mae_midgame', 0.0):.4f}` | < 0.2000 |\n")
            f.write(f"| **尾盘死区误差 (Endgame MAE)** | `{v_info.get('value_mae_endgame', 0.0):.4f}` | < 0.1500 |\n")
        if "tournament" in metrics:
            tour = metrics["tournament"]
            f.write(f"| **对抗得分率 (Score Rate)** | `{tour.get('score_rate_a', 0.0)*100:.1f}%` | > 52.0% 准予晋级 |\n")
            f.write(f"| **和棋率 (Draw Rate)** | `{tour.get('draw_rate', 0.0)*100:.1f}%` | 正常区间 (15% ~ 35%) |\n")
            f.write(f"| **重复局面和棋率 (Repetition)** | `{tour.get('repetition_rate', 0.0)*100:.1f}%` | 严格受控 (< 5.0%) |\n")
            f.write(f"| **平均对局手数 (Avg Plies)** | `{tour.get('avg_plies', 0.0):.1f}` | 80 ~ 180 手 |\n")
            f.write(f"| **相对 Elo 分差** | `{tour.get('elo_diff', 0.0):+.1f}` | > +25 Elo |\n")
        f.write("\n---\n")
