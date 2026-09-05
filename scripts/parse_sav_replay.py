#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 阶段工具: 军棋 App 对局记录 (.sav) 解码与记录提取工具。

支持将 .sav 二进制复盘文件解析为:
1. 人类可读的对局文本 (包含双方信息、初始底牌、逐步着法)
2. 结构化 JSON 格式 (供进一步分析、训练或可视化回放)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# 确保能引入项目核心模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import junqi.replay as r
    from junqi.rules import COLOR_CN, RANK_CN
    from junqi.state import GameState, Action
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

    # 兜底映射
    SAV_RANK_NAME = {
        1: "司令", 2: "军长", 3: "师长", 4: "旅长", 5: "团长",
        6: "营长", 7: "连长", 8: "排长", 9: "工兵",
        10: "炸弹", 11: "地雷", 12: "军旗"
    }

SPECIAL_EVENT = (255, 255, 1)


def cell_to_coord(cell: int) -> tuple[int, int]:
    """cell 索引 (0..59) 转为 (row, col) 坐标"""
    return (cell // 5, cell % 5)


def parse_raw_sav(path: str) -> dict:
    """纯标准库解析 .sav 二进制结构"""
    with open(path, "rb") as fh:
        data = fh.read()

    if len(data) < 99:
        raise ValueError(f"文件过小 ({len(data)} 字节)，非有效 .sav 复盘文件")

    version = data[0]
    seed1 = data[1]
    timestamp = struct.unpack_from("<I", bytes(data[2:5]) + b"\x00")[0] * 256
    name1 = data[5:16].split(b"\x00")[0].decode("gbk", errors="replace")
    score1 = struct.unpack_from("<H", data, 16)[0]
    name2 = data[18:29].split(b"\x00")[0].decode("gbk", errors="replace")
    score2 = struct.unpack_from("<H", data, 29)[0]
    flags = (data[31], data[32])
    n_moves = struct.unpack_from("<H", data, 33)[0]
    mode = data[35]
    ai_think = data[36]
    seed2 = data[37]
    ai_level = data[38]
    table = tuple(data[39:99])

    expected_len = 99 + 3 * n_moves
    if len(data) != expected_len:
        print(f"[WARN] 文件大小与步数标头不符: 实际 {len(data)}B, 标头预期 {expected_len}B")

    moves = []
    for i in range(n_moves):
        offset = 99 + 3 * i
        if offset + 3 <= len(data):
            moves.append(struct.unpack_from("<BBB", data, offset))

    return {
        "file": os.path.basename(path),
        "version": version,
        "seed1": seed1,
        "timestamp_approx": timestamp,
        "player1": {"name": name1, "rating": score1},
        "player2": {"name": name2, "rating": score2},
        "flags": flags,
        "mode": mode,
        "ai_think": ai_think,
        "ai_level": ai_level,
        "initial_table": list(table),
        "raw_moves": moves,
    }


def extract_game_details(path: str) -> dict:
    """结合引擎回放，提取完整的逐手动态与终局信息"""
    if not HAS_ENGINE:
        raw = parse_raw_sav(path)
        raw["note"] = "未加载 junqi 引擎，仅提供底层二进制字段"
        return raw

    game = r.parse_sav(path)
    board = r.board_from_table(game.table)
    st = GameState(board=board)

    moves_detail = []
    color_map = {"r": "红方", "b": "蓝方"}

    for i, (a, b, c) in enumerate(game.moves):
        seat = st.ply % 2
        player_name = game.names[seat]
        assigned_color = st.seat_color.get(seat)
        color_str = color_map.get(assigned_color, "未定色") if assigned_color else "未定色"

        step_record = {
            "index": i + 1,
            "ply": st.ply + 1,
            "seat": seat,
            "player": player_name,
            "player_color": color_str,
        }

        if (a, b, c) == SPECIAL_EVENT:
            step_record["type"] = "special_event"
            step_record["desc"] = "对局中止/特殊事件 (认输、求和或强退)"
            moves_detail.append(step_record)
            break

        if a == b and c == 1:
            pos = cell_to_coord(a)
            piece = st.board.get(pos)
            p_color = COLOR_CN.get(piece.color, "") if piece else ""
            p_rank = RANK_CN.get(piece.rank, "未知") if piece else "未知"
            piece_str = f"{p_color}{p_rank}"

            step_record["type"] = "flip"
            step_record["coord"] = list(pos)
            step_record["cell"] = a
            step_record["revealed_piece"] = piece_str
            step_record["desc"] = f"翻开 ({pos[0]},{pos[1]}) -> 【{piece_str}】"
            act = Action("flip", pos)

        elif a != b and c in (1, 3):
            frm = cell_to_coord(a)
            to = cell_to_coord(b)
            src_p = st.board.get(frm)
            dst_p = st.board.get(to)

            src_str = f"{COLOR_CN.get(src_p.color, '')}{RANK_CN.get(src_p.rank, '')}" if src_p else "未知"
            step_record["from"] = list(frm)
            step_record["to"] = list(to)
            step_record["piece"] = src_str

            if dst_p is None:
                step_record["type"] = "move"
                step_record["desc"] = f"【{src_str}】移动 ({frm[0]},{frm[1]}) -> ({to[0]},{to[1]})"
            elif c == 1:
                dst_str = f"{COLOR_CN.get(dst_p.color, '')}{RANK_CN.get(dst_p.rank, '')}"
                step_record["type"] = "capture"
                step_record["target_piece"] = dst_str
                step_record["desc"] = f"【{src_str}】在 ({to[0]},{to[1]}) 吃掉【{dst_str}】"
            else:
                dst_str = f"{COLOR_CN.get(dst_p.color, '')}{RANK_CN.get(dst_p.rank, '')}"
                step_record["type"] = "both_die"
                step_record["target_piece"] = dst_str
                step_record["desc"] = f"【{src_str}】与 ({to[0]},{to[1]}) 的【{dst_str}】同归于尽"

            act = Action("move", frm, to)
        else:
            step_record["type"] = "unknown"
            step_record["raw"] = [a, b, c]
            step_record["desc"] = f"未知动作码 ({a}, {b}, {c})"
            moves_detail.append(step_record)
            break

        moves_detail.append(step_record)
        st = st.apply(act)

    winner_desc = "未终局/中途退出"
    if st.winner == 0:
        winner_desc = f"{game.names[0]} (先手/玩家1) 胜 ({st.win_reason})"
    elif st.winner == 1:
        winner_desc = f"{game.names[1]} (后手/玩家2) 胜 ({st.win_reason})"
    elif st.winner == -1:
        winner_desc = f"和棋 ({st.win_reason})"

    return {
        "file": os.path.basename(path),
        "player1": {"name": game.names[0], "rating": game.ratings[0]},
        "player2": {"name": game.names[1], "rating": game.ratings[1]},
        "mode": game.mode,
        "total_moves": len(game.moves),
        "executed_plies": st.ply,
        "winner": st.winner,
        "win_reason": st.win_reason,
        "result_desc": winner_desc,
        "moves": moves_detail,
    }


def format_text_report(detail: dict) -> str:
    """生成整齐的对局文本记录"""
    lines = []
    lines.append("=" * 64)
    lines.append(f" 军棋对局记录提取: {detail['file']}")
    lines.append("=" * 64)
    p1 = detail["player1"]
    p2 = detail["player2"]
    lines.append(f"先手 (玩家1): {p1['name']} (积分: {p1['rating']})")
    lines.append(f"后手 (玩家2): {p2['name']} (积分: {p2['rating']})")
    lines.append(f"对局模式: {detail.get('mode', '未知')}")
    lines.append(f"总动作数: {detail.get('total_moves', 0)} 手 (实际推演: {detail.get('executed_plies', 0)} 手)")
    lines.append(f"判定结果: {detail.get('result_desc', '未知')}")
    lines.append("-" * 64)
    lines.append(" 逐步行动记录 (行 0-11, 列 0-4):")
    lines.append("-" * 64)

    for m in detail.get("moves", []):
        ply = m.get("ply", 0)
        player = m.get("player", "")
        p_col = m.get("player_color", "")
        desc = m.get("desc", "")
        lines.append(f"第 {ply:3d} 手 | {player} [{p_col}] {desc}")

    lines.append("=" * 64)
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="军棋 .sav 复盘文件提取工具")
    parser.add_argument("path", help=".sav 文件路径")
    parser.add_argument("--json", action="store_true", help="输出为 JSON 格式")
    parser.add_argument("--out", "-o", default=None, help="导出到指定文件路径 (.txt 或 .json)")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"[错误] 文件不存在: {args.path}", file=sys.stderr)
        sys.exit(1)

    details = extract_game_details(args.path)

    if args.json:
        content = json.dumps(details, ensure_ascii=False, indent=2)
    else:
        content = format_text_report(details)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[成功] 记录已导出至: {args.out}")
    else:
        print(content)


if __name__ == "__main__":
    main()

