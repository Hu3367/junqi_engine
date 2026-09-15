# ⚠️ 历史验证脚本（Archived / 不是测试）

> 标注日期：2026-09-15 ｜ 依据：[`reviews/CODE_REVIEW_2026-09-15.md`](../../reviews/CODE_REVIEW_2026-09-15.md) C10 / C12

## 这是什么

本目录与 `scripts/archive/` 内容重复，都是**一次性验证脚本**（同源文件）。它们：

- **不是 pytest 用例**：文件名不匹配 `pytest.ini` 的 `python_files = test_*.py`，
  所以 `pytest tests/` 根本不会收集它们——`tests/utils/` 里的“测试”不会给任何质量保证；
- 依赖 `junqi.train_rl` 的**内部**统计原语（`wilson_lower_bound`），
  而门控的唯一真源已统一为 `junqi/eval_gate.py::run_gate`；
- 其中若干与**已删除**的 `junqi/expert/` 包同期产出。

| 文件 | 当初用途 |
|---|---|
| `_verify_a2.py` | 验证 A2 估值增强项（行营势力/死区势能/暗子时差） |
| `_verify_fit_weights.py` | 验证复盘权重拟合结果 |
| `_verify_rebase_state.py` | 验证 `--rebase-baseline` 后的基线状态 |
| `_verify_root_cause.py` | 定位 Value 塌缩根因 |

## 为什么留在 `tests/` 下会被误读

放在 `tests/` 目录里会让人以为“这些是被执行过的测试”。实际执行的是
`tests/test_*.py`（约 40 个文件、400+ 用例）。本目录的脚本**一次都没有**在
测试运行中被执行过。如果你需要长期守门某个结论，请写成
`tests/test_*.py` 下的正式用例。

## 使用约定

1. 不要把这里的脚本当作可用接口或架构依据。需要能力请用现役入口：
   测试 → `pytest tests/`；评测 → `python -m junqi benchmark` / `python -m junqi gate`；
   数据集 → `python -m junqi export_dataset`。
2. 不要在它们之上继续加功能；需要长期守门就写成 `tests/test_*.py` 下的正式用例。
3. 若 `junqi/train_rl.py` 的重复统计原语将来被删除，这些脚本会 ImportError；
   届时请改用 `junqi/eval_gate.py` 的等价函数，或直接删除，**不要**为兼容它们保留死代码。
