"""复盘点评全量汇总与算法改进数据管线 (Review Summary Pipeline)。

实现用户核心需求：
一键汇总所有复盘的单步点评记录，将其转化为三大算法改进资产：
1. 【诊断分析报告】(reports/review_summary_report.md)：
   - 聚合统计失误类型分布 (自杀弃营、炸弹滥撞、盲目翻棋等) 与高频问题分布。
2. 【算法蒸馏与微调数据集】(datasets/user_review_labeled.json)：
   - 将人工纠偏局面与推荐动作转化为高置信度训练样本，可无缝输入 train_value_distill.py。
3. 【自动化回归测试用例套件】(tests/test_user_reviewed_tactics.py)：
   - 将用户点评批评的恶手与推荐好招固化为 pytest 自动化测试，杜绝模型再次重犯。
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .config import RuleConfig
from .review_storage import REVIEW_LABELS, ReviewStorage
from .rules import Rank
from .state import Action, GameState
from .tactical_sampler import dict_to_action, dict_to_state


class ReviewSummaryPipeline:
    """复盘点评全量汇总与算法改进引擎。"""

    def __init__(self, storage: Optional[ReviewStorage] = None):
        self.storage = storage or ReviewStorage()

    def run_all(
        self,
        out_report_path: str = "reports/review_summary_report.md",
        out_dataset_path: str = "datasets/user_review_labeled.json",
        out_test_path: str = "tests/test_user_reviewed_tactics.py",
    ) -> Dict[str, Any]:
        """执行全量汇总、导出数据集与回归测试生成。"""
        reviews = self.storage.list_reviews()
        if not reviews:
            return {
                "total_reviews": 0,
                "report_file": None,
                "dataset_file": None,
                "test_file": None,
                "summary": "当前点评库为空，暂无点评记录可汇总。",
            }

        # 1. 生成汇总报告
        report_content = self.generate_markdown_report(reviews)
        os.makedirs(os.path.dirname(out_report_path) or ".", exist_ok=True)
        with open(out_report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. 导出训练/蒸馏数据集
        dataset_records = self.export_labeled_dataset(reviews, out_dataset_path)

        # 3. 自动生成回归测试套件
        tests_generated = self.generate_regression_tests(reviews, out_test_path)

        label_counts = Counter(r.get("label", "other") for r in reviews)

        return {
            "total_reviews": len(reviews),
            "label_counts": dict(label_counts),
            "report_file": out_report_path,
            "dataset_file": out_dataset_path,
            "dataset_samples": len(dataset_records),
            "test_file": out_test_path,
            "tests_generated": tests_generated,
            "summary": f"已成功汇总 {len(reviews)} 条点评记录，生成训练样本 {len(dataset_records)} 条，自动化回归测试 {tests_generated} 个！",
        }

    def generate_markdown_report(self, reviews: List[dict]) -> str:
        """生成详细的 Markdown 复盘与战术失误诊断报告。"""
        total = len(reviews)
        games_set = {r.get("source_file") for r in reviews if r.get("source_file")}
        label_counter = Counter(r.get("label", "other") for r in reviews)

        lines = [
            "# 军棋复盘点评与战术诊断全量汇总报告",
            "",
            f"> 统计时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  |  点评总数：**{total}** 条  |  涵盖复盘：**{len(games_set)}** 局",
            "",
            "## 一、战术特征与失误分布统计",
            "",
            "| 战术分类 | 标签说明 | 样本数量 | 占比 | 算法改进指导原则 |",
            "| :--- | :--- | :---: | :---: | :--- |",
        ]

        strategy_guidelines = {
            "camp_abandon": "严禁行营内关键子力无故弃营，强化要塞驻守分与出营被扑杀惩罚",
            "bomb_abuse": "严格约束炸弹主动出击兑子，仅允许兑司令/军长/师长，严惩撞小子",
            "flip_risk": "空营附近无威胁时鼓励翻棋，面对敌方大子压境时严控盲目翻棋风险",
            "missed_kill": "吃旗与绝杀判定前置，强化 1-ply 杀局与行营单向扑杀优先级",
            "expert_blunder": "校准 Star1 静态搜索与攻防加权，抑制局部短视与伪死锁误判",
            "human_better": "提取人类玩家大局观模式，注入先验 Policy 蒸馏数据集",
            "good_move": "正向强化标杆走法，提升对应落点候选价值",
            "dubious": "纳入重点推演题库，通过 5 层超深搜进一步量化验算",
            "other": "战术心得持续沉淀与归纳",
        }

        for k, name in REVIEW_LABELS.items():
            cnt = label_counter.get(k, 0)
            pct = (cnt / total * 100.0) if total > 0 else 0.0
            guide = strategy_guidelines.get(k, "-")
            lines.append(f"| `{k}` | {name} | **{cnt}** | {pct:.1f}% | {guide} |")

        lines.extend([
            "",
            "---",
            "",
            "## 二、高价值纠偏与重点点评案例清单",
            "",
            "以下整理了所有标记为恶手、人类更优或带有明确纠正走法的案例，已直接同步导出为算法训练集与自动化回归用例：",
            "",
        ])

        for i, r in enumerate(reviews, 1):
            lbl_name = r.get("label_name", r.get("label", ""))
            src = r.get("source_file", "")
            ply = r.get("ply", 0)
            played = r.get("action_played_desc", str(r.get("action_played")))
            rec = r.get("action_recommended_desc", str(r.get("action_recommended", "")))
            cmt = r.get("comment", "").strip() or "（未填写文字说明）"

            lines.append(f"### 案例 #{i}：[{lbl_name}] · 《{src}》第 {ply} 步")
            lines.append(f"- **行动方**：座位 {r.get('turn_seat')} ({r.get('turn_color', '')})")
            lines.append(f"- **实战走法**：`{played}`")
            if rec:
                lines.append(f"- **人工推荐纠正**：`{rec}` ⭐")
            lines.append(f"- **点评分析**：{cmt}")
            lines.append(f"- **时间**：{r.get('updated_at', r.get('created_at', ''))}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 三、算法自动化落地行动项")
        lines.append("1. **价值与策略蒸馏**：运行 `python -m junqi train_value_distill --data datasets/user_review_labeled.json`，将上述纠偏样本直接微调注入神经网络；")
        lines.append("2. **自动化防线守护**：运行 `pytest tests/test_user_reviewed_tactics.py`，验证当前专家引擎和混合引擎是否已成功纠正上述历史恶手。")
        lines.append("")

        return "\n".join(lines)

    def export_labeled_dataset(self, reviews: List[dict], out_path: str) -> List[dict]:
        """将点评记录导出为与 distill_tactical_labeled.json 兼容的高质量打标数据集。"""
        dataset: List[dict] = []

        for r in reviews:
            state_dict = r.get("state_dict")
            if not state_dict:
                continue

            played_dict = r.get("action_played")
            rec_dict = r.get("action_recommended")
            label = r.get("label", "other")
            comment = r.get("comment", "")

            # 映射为打标体系裁决
            if label in ("expert_blunder", "camp_abandon", "bomb_abuse", "flip_risk", "missed_kill"):
                judgment = "expert_blunder"
            elif label == "human_better":
                judgment = "human_better"
            elif label == "good_move":
                judgment = "both_fine"
            else:
                judgment = "skip"

            sample = {
                "game_id": r.get("source_file", "unknown"),
                "target_ply": r.get("ply", 0),
                "state": state_dict,
                "human_move": rec_dict or played_dict,
                "expert_move": played_dict,
                "expert_score": -100.0 if judgment == "expert_blunder" else 50.0,
                "action_judgment": judgment,
                "comment": comment,
                "label_meta": {
                    "raw_label": label,
                    "action_recommended": rec_dict,
                    "action_played": played_dict,
                },
                "verified": True,
                "timestamp": r.get("updated_at", r.get("created_at", "")),
            }
            dataset.append(sample)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        return dataset

    def generate_regression_tests(self, reviews: List[dict], out_path: str) -> int:
        """根据点评中的恶手纠正案例，自动生成 pytest 自动化回归测试代码。"""
        actionable_reviews = [
            r for r in reviews
            if r.get("state_dict") and (
                r.get("label") in ("expert_blunder", "camp_abandon", "bomb_abuse", "missed_kill", "human_better")
                or r.get("action_recommended")
            )
        ]

        if not actionable_reviews:
            # 即使没有可执行的恶手，也生成一个空的合规测试桩
            code = (
                '"""自动生成的复盘点评战术回归测试套件。"""\n'
                'import pytest\n\n'
                'def test_no_blunders_registered():\n'
                '    """当前尚未登记明确战术恶手纠正案例。"""\n'
                '    pass\n'
            )
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
            return 0

        lines = [
            '"""自动生成的复盘点评战术回归测试套件。',
            '',
            '本文件由 ReviewSummaryPipeline 自动提取用户历史复盘点评与恶手案例生成。',
            '确保引擎算法升级后，永不再犯历史复盘中人工批评过的恶手！',
            '"""',
            'from __future__ import annotations',
            '',
            'import pytest',
            'from junqi.ai import ExpertAgent',
            'from junqi.config import SearchConfig',
            'from junqi.state import Action',
            'from junqi.tactical_sampler import dict_to_action, dict_to_state',
            '',
            '',
        ]

        for i, r in enumerate(actionable_reviews):
            safe_name = f"test_review_case_{i+1}_{r.get('ply', 0)}_{r.get('label', 'blunder')}"
            played_dict = r.get("action_played")
            rec_dict = r.get("action_recommended")
            cmt = r.get("comment", "").replace('"', '\\"').replace("\n", " ")
            src = r.get("source_file", "")
            ply = r.get("ply", 0)

            # 序列化为内联 json 字符串
            st_json_str = json.dumps(r["state_dict"], ensure_ascii=False)

            lines.append(f'def {safe_name}():')
            lines.append(f'    """案例 #{i+1}：来自《{src}》第 {ply} 步点评。')
            lines.append(f'    点评备注：{cmt}')
            lines.append('    """')
            lines.append(f'    state_data = {st_json_str}')
            lines.append('    st = dict_to_state(state_data)')
            lines.append('    agent = ExpertAgent(search=SearchConfig(depth=3), seed=42)')
            lines.append('    chosen = agent.select_action(st)')
            lines.append('')

            if played_dict:
                p_kind = repr(played_dict.get("kind"))
                p_frm = tuple(played_dict.get("frm"))
                p_to = tuple(played_dict.get("to")) if played_dict.get("to") else None
                lines.append(f'    blunder = Action({p_kind}, {p_frm}, {p_to})')
                lines.append('    # 严禁引擎重新选择被标记为恶手的走法')
                lines.append('    assert chosen != blunder, f"引擎重新走出了历史恶手: {chosen}"')

            if rec_dict:
                r_kind = repr(rec_dict.get("kind"))
                r_frm = tuple(rec_dict.get("frm"))
                r_to = tuple(rec_dict.get("to")) if rec_dict.get("to") else None
                lines.append(f'    recommended = Action({r_kind}, {r_frm}, {r_to})')
                lines.append('    top_acts = [a for a, _ in agent.choose_actions(st, topn=3)]')
                lines.append('    # 推荐正着应位于前列候选')
                lines.append('    assert recommended in top_acts, f"人工推荐正着 {recommended} 未出现在 Top3 走法中: {top_acts}"')

            lines.append('')
            lines.append('')

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return len(actionable_reviews)
