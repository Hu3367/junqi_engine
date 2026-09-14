# ⚠️ `junqi/expert/` 已废弃（Deprecated）

> 判定日期：2026-09-15 ｜ 依据：[`reviews/CODE_REVIEW_2026-09-15.md`](../../reviews/CODE_REVIEW_2026-09-15.md) §R4

## 状态

**本包不可导入，且不在任何主流程上。** 请勿在此基础上继续开发。

```console
$ python -c "import junqi.expert"
NameError: name 'TacticalReport' is not defined
```

`junqi/expert/__init__.py` 的导入链在 `move_adapter` → `tactical_analyzer` 处就崩在
一个未定义的类型名上；即使绕过这一步，包内 10/11 个模块还引用了一批在当前
`GameState` / `RuleConfig` 上**根本不存在**的成员：

| 被引用 | 现状 |
|---|---|
| `state.Move` | `junqi/state.py` 只有 `Action`，从未定义 `Move` |
| `state.get_piece_at()` | 不存在 |
| `state.current_turn` / `turn_count` / `get_pieces()` | 不存在（对应字段是 `turn` / `ply`，访问器需另行实现） |
| `config.PIECE_RANKS` | 不存在（口径见 `config.EvalWeights.piece`） |
| `board.board.is_my_base` | 不存在（对应概念是 `rules.is_hq`） |

此外 `expert_engine.py:305` 的 `self._score_move(m, None, None, None, None, None)`
会把 `None` 传进 `_score_tempo_impact`，触发 AttributeError。

## 影响范围

**零。** 依赖图分析（AST 全量扫描）结果：`junqi/` 全库除本包内部自引用外，
没有任何 `from junqi.expert ...` / `from .expert ...` 的调用点；GUI、训练、
评估、CLI 全部不受影响。项目的在线传统搜索引擎是 **`junqi/search.py` 的
`ExpertSearchEngine`**（并由 `junqi/ai.py` / `junqi/hybrid_engine.py` 复用），
本包是与它并行、后被放弃的第三套实现。

README / 文档中出现的"传统搜索专家模型（ExpertSearchEngine）深度对齐"指的是
`junqi/search.py`，**不是**本包。

## 处置建议

按成本从低到高：

1. **维持废弃**（当前选择）：包就地保留作为历史参考，本文件说明原因。
   新增的 `tests/test_p0_expert_deprecated.py` 会守住"没有主流程依赖它"这条线。
2. **彻底删除**：`git rm -r junqi/expert/`（若确定不再需要从其中恢复思路）。
3. **复活**：需要补一个 `Move ↔ Action` 适配层（`move_adapter.py` 有雏形但未接通），
   并在 `GameState` 上补齐 `get_piece_at` 等门面方法；工作量接近重写，
   且会与 `junqi/search.py` 形成第二套引擎，不建议。

## 若坚持要在本包上开发

最小前置 gates：

1. 修 `tactical_analyzer.py` 的 `TacticalReport` 未定义；
2. 建立 `Move` 数据类并完成与 `Action` 的双向转换；
3. 为 `get_piece_at` / `current_turn` / `turn_count` / `get_pieces` / `is_my_base`
   提供实现或在 `GameState` 上补齐；
4. `python -c "import junqi.expert"` 必须通过，并补一个 import smoke test。
