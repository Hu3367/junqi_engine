# 已删除：`junqi/expert/` 实验性专家包（11 个模块）

> 处置日期：2026-09-15 ｜ 依据：[`reviews/CODE_REVIEW_2026-09-15.md`](../../reviews/CODE_REVIEW_2026-09-15.md) §R4
> 状态：**已从仓库删除**（`git rm -r junqi/expert/`）

## 为什么删

该包**不可导入**，且不在任何主流程上。删除前的实测：

```console
$ python -c "import junqi.expert"
NameError: name 'TacticalReport' is not defined
```

`junqi/expert/__init__.py` 的导入链在 `move_adapter` → `tactical_analyzer` 处崩在一个
未定义的类型名上。即使绕过这一步，包内 10/11 个模块还引用了一批在当前
`GameState` / `RuleConfig` 上**根本不存在**的成员：

| 被引用 | 现状 |
|---|---|
| `state.Move` | `junqi/state.py` 只有 `Action`，从未定义 `Move` |
| `state.get_piece_at()` | 不存在 |
| `state.current_turn` / `turn_count` / `get_pieces()` | 不存在（字段是 `turn` / `ply`） |
| `config.PIECE_RANKS` | 不存在（口径见 `config.EvalWeights.piece`） |
| `board.board.is_my_base` | 不存在（对应概念是 `rules.is_hq`） |

`expert_engine.py` 的 `_score_move(m, None, None, None, None, None)` 还会把 `None`
传进 `_score_tempo_impact`，触发 AttributeError。

AST 依赖图核验：`junqi/` 全库除本包内部自引用外无任何调用点（fan-in = 0）。
**项目的在线传统搜索引擎是 `junqi/search.py::ExpertSearchEngine`**，由
`junqi/ai.py` / `junqi/hybrid_engine.py` 复用——本包是与它并行、后被放弃的第三套实现。
README 里"传统搜索专家模型（ExpertSearchEngine）深度对齐"指的也是 `search.py`，**不是**本包。

## 被删除的文件

```
junqi/expert/__init__.py            junqi/expert/expert_engine.py
junqi/expert/conditional_value.py   junqi/expert/hidden_piece_belief.py
junqi/expert/mobility_calculator.py junqi/expert/move_adapter.py
junqi/expert/rule_validator.py      junqi/expert/search_optimizer.py
junqi/expert/tactical_analyzer.py   junqi/expert/tempo_tracker.py
junqi/expert/threat_detection.py
```

连带删除：`scripts/test_expert_core.py`（唯一引用该包的脚本，367 行）。

## 如何找回

文件都在版本控制内，可直接取回：

```bash
git log --oneline -- junqi/expert                # 找到删除前的提交
git checkout <commit>^ -- junqi/expert           # 恢复整个包
git checkout <commit>^ -- scripts/test_expert_core.py
```

## 若将来要复活

最小前置 gates（工作量接近重写，且会与 `junqi/search.py` 形成第二套引擎，不建议）：

1. 修 `tactical_analyzer.py` 的 `TacticalReport` 未定义；
2. 建立 `Move` 数据类并完成与 `Action` 的双向转换（原 `move_adapter.py` 只有雏形）；
3. 为 `get_piece_at` / `current_turn` / `turn_count` / `get_pieces` / `is_my_base`
   提供实现，或在 `GameState` 上补齐门面方法；
4. `python -c "import junqi.expert"` 必须通过，并补 import smoke test；
5. 明确它与 `junqi/search.py` 的职责边界，避免两套引擎并行。

## 遗留引用

`reports/expert_engine_v1/` 下的历史报告（`EXPERT_ENGINE_SUMMARY.md`、
`EXPERT_ENGINE_V1_0_COMPLETE_FINAL.md`、`EXPERT_ENGINE_V1_0_FINAL_REPORT.md`）中
仍有 `from junqi.expert import ExpertEngine` 的示例。那些是**该包当初的交付报告**，
属于历史存档，不回改；请以本文件为准，不要把其中的用法当作可用 API。
