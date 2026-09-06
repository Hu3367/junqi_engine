"""复盘点评数据持久化存储与管理模块 (Review Storage)。

负责将用户对复盘局面的单步战术点评、定性评级 (Label) 与推荐走法 (Recommended Action)
进行持久化存储、按盘面与步数检索、增删改查。
数据统一保存在 reviews/annotations.json 中。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from .state import Action, GameState, position_key
from .tactical_sampler import action_to_dict, dict_to_action, state_to_dict

DEFAULT_STORAGE_PATH = "reviews/annotations.json"

# 预设点评标签体系
REVIEW_LABELS = {
    "expert_blunder": "❌ 专家恶手 (战术硬伤)",
    "camp_abandon": "⚠️ 自杀弃营 (拱手让要塞)",
    "bomb_abuse": "💣 炸弹乱撞 (撞小子/贱卖)",
    "flip_risk": "🎲 盲目翻棋 (高风险赌博)",
    "missed_kill": "🎯 错失绝杀 (漏吃大子/旗)",
    "human_better": "👤 人类更优 (专家短视)",
    "good_move": "⭐ 卓越正着 (典范好棋)",
    "dubious": "❓ 存疑试探 (有待商榷)",
    "other": "📝 其他战术心得",
}


class ReviewStorage:
    """复盘点评持久化存储管理器。"""

    def __init__(self, storage_path: str = DEFAULT_STORAGE_PATH):
        self.storage_path = storage_path
        self._ensure_dir()
        self._data: Dict[str, dict] = self._load()

    def _ensure_dir(self):
        dir_path = os.path.dirname(self.storage_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def _load(self) -> Dict[str, dict]:
        if not os.path.exists(self.storage_path):
            return {}
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        self._ensure_dir()
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def make_key(source_file: str, ply: int) -> str:
        base = os.path.basename(source_file)
        return f"{base}:{ply}"

    def save_review(
        self,
        source_file: str,
        ply: int,
        state: GameState,
        action_played: Action,
        label: str,
        comment: str,
        action_recommended: Optional[Action] = None,
        action_played_desc: str = "",
        action_recommended_desc: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        """保存或更新单步点评记录。"""
        key = self.make_key(source_file, ply)
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        existing = self._data.get(key, {})
        created_at = existing.get("created_at", now_str)

        label_name = REVIEW_LABELS.get(label, label)

        rec = {
            "review_id": existing.get("review_id", f"rev_{int(time.time()*1000)}_{ply}"),
            "source_file": os.path.basename(source_file),
            "ply": ply,
            "turn_seat": state.turn,
            "turn_color": state.my_color(),
            "position_key": position_key(state),
            "action_played": action_to_dict(action_played),
            "action_played_desc": action_played_desc or str(action_played),
            "action_recommended": action_to_dict(action_recommended) if action_recommended else None,
            "action_recommended_desc": action_recommended_desc or (str(action_recommended) if action_recommended else ""),
            "label": label,
            "label_name": label_name,
            "comment": comment.strip(),
            "state_dict": state_to_dict(state),
            "metadata": metadata or {},
            "created_at": created_at,
            "updated_at": now_str,
        }

        self._data[key] = rec
        self._save()
        return rec

    def get_review(self, source_file: str, ply: int) -> Optional[dict]:
        """获取指定文件与步数的点评。"""
        key = self.make_key(source_file, ply)
        return self._data.get(key)

    def delete_review(self, source_file: str, ply: int) -> bool:
        """删除指定点评。"""
        key = self.make_key(source_file, ply)
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def list_reviews(self, source_file: Optional[str] = None) -> List[dict]:
        """按时间倒序列出点评。支持按源文件过滤。"""
        if source_file:
            base = os.path.basename(source_file)
            res = [v for k, v in self._data.items() if v.get("source_file") == base]
        else:
            res = list(self._data.values())
        res.sort(key=lambda r: (r.get("source_file", ""), r.get("ply", 0)))
        return res

    def count(self, source_file: Optional[str] = None) -> int:
        """获取点评总条数。"""
        return len(self.list_reviews(source_file))
