"""人类高手复盘“分歧率与胜率断崖”自动化扫描工具 (Replay Divergence & Cliff Scanner)。

自动读取真实人类对局（军旗复盘/*.sav），逐手回放并比对：
1. 人类高手着法在 AI 评估体系中的排名 (Top-1 / Top-3 吻合度) 与分歧分值；
2. 筛选重大战术分歧点 (Divergence: 人类妙手被 AI 严重低估)；
3. 借助全盘真值演进探测胜率断崖 (Valuation Cliff: AI 首选动作是否存在严重致命盲区)。

用法示例:
    python scripts/scan_replays.py --max-games 10 --replay-dir "军旗复盘"
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from junqi.ai import evaluate_expert
from junqi.config import EvalWeights, RuleConfig
from junqi.hybrid_engine import HybridDecisionEngine
from junqi.replay import SPECIAL_EVENT, board_from_table, cell_rc, parse_sav
from junqi.rules import RANK_CN, Rank, other
from junqi.state import Action, GameState, position_key


def format_action_cn(st: GameState, a: Action) -> str:
    """将动作转为高可读性中文描述。"""
    if a.kind == "flip":
        pc = st.board.get(a.frm)
        return f"翻({a.frm[0]},{a.frm[1]})" + (f"[{pc.color}{RANK_CN[pc.rank]}]" if pc and pc.revealed else "")
    mover = st.board.get(a.frm)
    target = st.board.get(a.to)
    mover_str = f"{mover.color}{RANK_CN[mover.rank]}" if mover and mover.revealed else "暗子"
    if target is None:
        return f"走({a.frm[0]},{a.frm[1]})->({a.to[0]},{a.to[1]})[{mover_str}占空地]"
    tgt_str = f"{target.color}{RANK_CN[target.rank]}" if target.revealed else "暗子"
    return f"走({a.frm[0]},{a.frm[1]})->({a.to[0]},{a.to[1]})[{mover_str}吃{tgt_str}]"


def scan_replays(
    replay_dir: str = "军旗复盘",
    max_games: int = 10,
    output_dir: str = "reports/replays",
    k_worlds: int = 2,
    device: str = "cpu",
    diff_threshold: float = 20.0,
    exclude_flag_toying: bool = True,
) -> Tuple[str, str]:
    """主扫描管线。"""
    os.makedirs(output_dir, exist_ok=True)
    cfg = RuleConfig()
    engine = HybridDecisionEngine(k_worlds=k_worlds, device=device, seed=42)

    sav_files = sorted(glob.glob(os.path.join(replay_dir, "*.sav")))
    if not sav_files:
        print(f"[ERROR] 未在目录 '{replay_dir}' 找到任何 .sav 复盘文件！")
        return "", ""

    if max_games > 0:
        sav_files = sav_files[:max_games]

    print(f"============================================================")
    print(f" [START] 启动人类高手复盘分歧率与胜率断崖扫描 (Replay Scanner)")
    print(f" 待扫描对局: {len(sav_files)} 局 | 采样世界数: {k_worlds} | 判定阈值: {diff_threshold}")
    print(f" 排除故意不捉旗戏耍: {'开启' if exclude_flag_toying else '关闭'}")
    print(f" 输出目录: {output_dir}")
    print(f"============================================================\n")

    start_time = time.time()
    total_plies = 0
    top1_matches = 0
    top3_matches = 0
    flag_toying_plies = 0
    effective_plies = 0
    effective_top1 = 0
    effective_top3 = 0

    all_divergences: List[dict] = []
    all_cliffs: List[dict] = []

    for g_idx, fpath in enumerate(sav_files, 1):
        fname = os.path.basename(fpath)
        try:
            game = parse_sav(fpath)
            board = board_from_table(game.table)
        except Exception as e:
            print(f"[{g_idx:02d}/{len(sav_files):02d}] 跳过非法文件 {fname}: {e}")
            continue

        st = GameState(board=board, cfg=cfg)
        game_plies = 0
        game_divergences = 0
        game_cliffs = 0

        for i, (a, b, c) in enumerate(game.moves):
            if (a, b, c) == SPECIAL_EVENT:
                break
            if a == b and c == 1:
                act_human = Action("flip", cell_rc(a))
            elif a != b and c in (1, 3):
                act_human = Action("move", cell_rc(a), cell_rc(b))
            else:
                break

            legal_acts = st.legal_actions()
            if act_human not in legal_acts:
                break

            total_plies += 1
            game_plies += 1

            # 调用引擎评估
            scored = engine.choose_actions(st, topn=len(legal_acts))
            if not scored:
                st = st.apply(act_human)
                continue

            act_ai, best_score = scored[0]

            # 寻找人类动作在 AI 评估中的位次
            human_rank = None
            human_score = -9999.0
            for rk_idx, (cand_a, cand_s) in enumerate(scored, 1):
                if cand_a == act_human:
                    human_rank = rk_idx
                    human_score = cand_s
                    break

            if act_human == act_ai:
                top1_matches += 1
            if human_rank is not None and human_rank <= 3:
                top3_matches += 1

            # 检查是否为“AI 推荐吃军旗直接制胜，但人类故意不捉旗戏耍对手”的情形
            ai_takes_flag = (
                act_ai.kind == "move" and
                (ai_tgt := st.board.get(act_ai.to)) is not None and
                ai_tgt.revealed and ai_tgt.rank == Rank.QI
            )
            human_takes_flag = (
                act_human.kind == "move" and
                (human_tgt := st.board.get(act_human.to)) is not None and
                human_tgt.revealed and human_tgt.rank == Rank.QI
            )
            is_flag_toying = ai_takes_flag and not human_takes_flag

            if is_flag_toying:
                flag_toying_plies += 1

            if not is_flag_toying or not exclude_flag_toying:
                effective_plies += 1
                if act_human == act_ai:
                    effective_top1 += 1
                if human_rank is not None and human_rank <= 3:
                    effective_top3 += 1

            score_gap = best_score - human_score if human_rank is not None else 999.0

            # 1. 战术分歧点探针 (Top-3 之外或分值落后 > diff_threshold)
            # 排除人类故意不捉旗戏耍对手的垃圾时间
            if is_flag_toying and exclude_flag_toying:
                pass  # 戏耍对手不吃军旗，不计入战术分歧
            else:
                is_divergence = (human_rank is None or human_rank > 3 or score_gap >= diff_threshold)
                if is_divergence:
                    game_divergences += 1
                    all_divergences.append({
                        "game_id": g_idx,
                        "filename": fname,
                        "players": list(game.names),
                        "ply": st.ply,
                        "seat": st.turn,
                        "act_human": str(act_human),
                        "act_human_desc": format_action_cn(st, act_human),
                        "human_rank": human_rank,
                        "human_score": round(human_score, 2),
                        "act_ai": str(act_ai),
                        "act_ai_desc": format_action_cn(st, act_ai),
                        "ai_score": round(best_score, 2),
                        "score_gap": round(score_gap, 2),
                    })

            # 2. 胜率断崖探针 (对比 AI 动作与人类动作在真实棋盘演进下的专家估值)
            # 若 AI 推荐的动作导致己方估值剧烈下跌 (落后人类动作 > 35.0 分)
            try:
                st_human_next = st.apply(act_human)
                eval_human = evaluate_expert(st_human_next, my=st.my_color())

                st_ai_next = st.apply(act_ai)
                eval_ai = evaluate_expert(st_ai_next, my=st.my_color())

                val_drop = eval_human - eval_ai
                if val_drop >= 35.0:
                    game_cliffs += 1
                    all_cliffs.append({
                        "game_id": g_idx,
                        "filename": fname,
                        "ply": st.ply,
                        "seat": st.turn,
                        "act_ai": str(act_ai),
                        "act_ai_desc": format_action_cn(st, act_ai),
                        "act_human": str(act_human),
                        "act_human_desc": format_action_cn(st, act_human),
                        "eval_human": round(eval_human, 2),
                        "eval_ai": round(eval_ai, 2),
                        "val_drop": round(val_drop, 2),
                        "description": (
                            f"AI 推荐动作 {format_action_cn(st, act_ai)} 导致客观估值断崖下跌 {val_drop:.1f} 分，"
                            f"人类高手选择 {format_action_cn(st, act_human)} 避开陷阱！"
                        )
                    })
            except Exception:
                pass

            # 执行人类着法继续演进
            st = st.apply(act_human)

        print(f"[{g_idx:02d}/{len(sav_files):02d}] {fname[:25]}... | "
              f"手数: {game_plies:02d} | 分歧点: {game_divergences:02d} | 断崖点: {game_cliffs:02d}")

    total_elapsed = time.time() - start_time
    top1_rate = (top1_matches / total_plies * 100.0) if total_plies > 0 else 0.0
    top3_rate = (top3_matches / total_plies * 100.0) if total_plies > 0 else 0.0
    eff_top1_rate = (effective_top1 / effective_plies * 100.0) if effective_plies > 0 else 0.0
    eff_top3_rate = (effective_top3 / effective_plies * 100.0) if effective_plies > 0 else 0.0

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 导出 JSON 数据集
    json_path = os.path.join(output_dir, f"replay_blunders_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": ts,
            "total_games": len(sav_files),
            "total_plies": total_plies,
            "flag_toying_plies": flag_toying_plies,
            "effective_plies": effective_plies,
            "top1_matches": top1_matches,
            "top1_rate_percent": round(top1_rate, 2),
            "top3_matches": top3_matches,
            "top3_rate_percent": round(top3_rate, 2),
            "effective_top1_matches": effective_top1,
            "effective_top1_rate_percent": round(eff_top1_rate, 2),
            "effective_top3_matches": effective_top3,
            "effective_top3_rate_percent": round(eff_top3_rate, 2),
            "divergences_count": len(all_divergences),
            "cliffs_count": len(all_cliffs),
            "divergences": all_divergences,
            "cliffs": all_cliffs,
        }, f, ensure_ascii=False, indent=2)

    # 导出 Markdown 报告
    md_path = os.path.join(output_dir, f"replay_divergence_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 人类高手复盘分歧率与胜率断崖扫描诊断报告\n\n")
        f.write(f"- **生成时间**：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **扫描对局数**：`{len(sav_files)}` 局真实对局\n")
        f.write(f"- **总评估手数**：`{total_plies}` 手（有效竞技手数: `{effective_plies}` 手，排除戏耍故意不夺旗: `{flag_toying_plies}` 手）\n")
        f.write(f"- **总耗时**：{total_elapsed:.2f} 秒（平均 {total_elapsed/max(1, len(sav_files)):.2f} 秒/局）\n\n")

        f.write(f"## 一、 人机吻合度与全局指标\n\n")
        f.write(f"| 指标项目 | 数值 | 说明 |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| **总评估手数** | **{total_plies}** | 复盘对局中的总行棋手数 |\n")
        f.write(f"| **过滤戏耍手数** | **{flag_toying_plies}** | 人类在可一步夺旗获胜时故意闲走不捉旗的手数（已成功排除） |\n")
        f.write(f"| **有效竞技手数** | **{effective_plies}** | 剔除垃圾时间后的真实战术较量手数 |\n")
        f.write(f"| **有效 Top-1 吻合率** | **{eff_top1_rate:.2f}%** | 排除戏耍后的真实 AI-人类高手首选手吻合度 |\n")
        f.write(f"| **有效 Top-3 吻合率** | **{eff_top3_rate:.2f}%** | 排除戏耍后的真实进入前三位比例 |\n")
        f.write(f"| **战术分歧点总数** | **{len(all_divergences)}** | 真实战术分歧（已彻底滤除夺旗戏耍噪音） |\n")
        f.write(f"| **胜率断崖点总数** | **{len(all_cliffs)}** | AI 推荐动作导致客观估值断崖下挫 > 35.0 分 |\n\n")

        f.write(f"## 二、 Top 致命胜率断崖案例 (AI 推荐走入陷阱 / 人类明智避开)\n\n")
        if not all_cliffs:
            f.write(f"> 🎉 本次扫描未发现明显的胜率断崖点，AI 推荐动作在完全信息下均保持稳健！\n\n")
        else:
            # 按估值暴跌幅度排序展示前 8 个案例
            top_cliffs = sorted(all_cliffs, key=lambda x: x["val_drop"], reverse=True)[:8]
            for idx, c in enumerate(top_cliffs, 1):
                f.write(f"### 案例 {idx}: {c['filename']} (第 {c['ply']} 手)\n\n")
                f.write(f"- **断崖幅度**：客观战力下跌 **-{c['val_drop']:.1f}** 分\n")
                f.write(f"- **人类高手选择**：`{c['act_human_desc']}`（实盘估值: `{c['eval_human']:.1f}`）\n")
                f.write(f"- **AI 推荐走法**：`{c['act_ai_desc']}`（推演估值: `{c['eval_ai']:.1f}`）\n")
                f.write(f"- **诊断结论**：{c['description']}\n\n")

        f.write(f"## 三、 典型战术分歧案例复盘 (人类高手高阶妙手 vs AI 低估)\n\n")
        if not all_divergences:
            f.write(f"> 本次扫描未发现显著战术分歧点。\n\n")
        else:
            top_divs = sorted(all_divergences, key=lambda x: x["score_gap"], reverse=True)[:8]
            for idx, d in enumerate(top_divs, 1):
                f.write(f"### 分歧 {idx}: {d['filename']} (第 {d['ply']} 手)\n\n")
                f.write(f"- **玩家**：{d['players'][0]} vs {d['players'][1]}\n")
                f.write(f"- **人类高手着法**：`{d['act_human_desc']}` (AI 评为第 {d['human_rank']} 位，得分: {d['human_score']})\n")
                f.write(f"- **AI 首选动作**：`{d['act_ai_desc']}` (得分: {d['ai_score']}, 领先分差: {d['score_gap']})\n\n")

        f.write(f"## 四、 结论与后续训练建议\n\n")
        f.write(f"1. **断崖盲区入库**：所有断崖案例均已写入 `{json_path}`，可直接转为测试用例；\n")
        f.write(f"2. **行为克隆蒸馏 (P3)**：有效 Top-1 吻合率为 {eff_top1_rate:.1f}%，通过将人类高手分歧动作作为高质量正样本强化训练，可进一步提升中盘复杂战术的胜率。\n")

    print(f"\n============================================================")
    print(f" [DONE] 扫描完成！总手数: {total_plies} (有效手数: {effective_plies}, 排除戏耍故意不捉旗: {flag_toying_plies})")
    print(f"  - 有效 Top-1 吻合率: {eff_top1_rate:.2f}% | 有效 Top-3 吻合率: {eff_top3_rate:.2f}%")
    print(f"  - 发现真实战术分歧点: {len(all_divergences)} 处 | 发现胜率断崖点: {len(all_cliffs)} 处")
    print(f" [OUTPUT] 错题库已保存: {json_path}")
    print(f" [REPORT] 诊断报告已生成: {md_path}")
    print(f"============================================================\n")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="人类高手复盘分歧率与胜率断崖扫描工具")
    parser.add_argument("--replay-dir", type=str, default="军旗复盘", help="复盘文件目录")
    parser.add_argument("--max-games", type=int, default=10, help="扫描最大对局数 (默认 10)")
    parser.add_argument("--output-dir", type=str, default="reports/replays", help="输出报告目录")
    parser.add_argument("--k-worlds", type=int, default=2, help="PIMC 采样世界数")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备")
    parser.add_argument("--diff-threshold", type=float, default=20.0, help="分歧分值告警阈值")
    parser.add_argument("--no-exclude-flag-toying", dest="exclude_flag_toying", action="store_false",
                        help="不排除人类故意不捉旗戏耍对手的手数")

    args = parser.parse_args()
    scan_replays(
        replay_dir=args.replay_dir,
        max_games=args.max_games,
        output_dir=args.output_dir,
        k_worlds=args.k_worlds,
        device=args.device,
        diff_threshold=args.diff_threshold,
        exclude_flag_toying=args.exclude_flag_toying,
    )


if __name__ == "__main__":
    main()
