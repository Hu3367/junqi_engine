# ⚠️ 历史归档脚本（Archived / 不作为 API）

> 标注日期：2026-09-15 ｜ 依据：[`reviews/CODE_REVIEW_2026-09-15.md`](../../reviews/CODE_REVIEW_2026-09-15.md) C10 / C12

## 这是什么

`scripts/archive/` 存放的是**一次性验证脚本**——当初用来复现某个具体结论后就被留在仓库里。
它们：

- **没有测试覆盖**，也不在 CI/`run_tests.bat` 的路径上；
- 依赖 `junqi.train_rl` 的**内部**统计原语（`wilson_lower_bound`、`decide_promotion`），
  这些原语已不再是门控的唯一真源（训练循环已统一委托 `junqi/eval_gate.py::run_gate`）；
- 其中若干脚本与**已删除**的 `junqi/expert/` 包同期产出，请勿据此推断当前架构。

| 文件 | 当初用途 | 现状 |
|---|---|---|
| `_verify_a2.py` | 验证 A2 估值增强项（行营势力/死区势能/暗子时差） | 一次性证据，已并入 `config.EvalWeights` |
| `_verify_fit_weights.py` | 验证复盘权重拟合结果 | 已被 `python -m junqi fit` 取代 |
| `_verify_rebase_state.py` | 验证 `--rebase-baseline` 后的基线状态 | 已被 `tests/test_p0_hotstart_and_opponent.py` 取代 |
| `_verify_root_cause.py` | 定位 Value 塌缩根因 | 结论已写入 `docs/CHANGELOG.md` |
| `benchmark_expert.py` / `benchmark_p2.py` | 早期专家引擎/BC 基准 | 已被 `python -m junqi benchmark` 取代 |
| `gen_assets.py` / `gen_eval_sets.py` | 资源与题库生成 | 现役版本在 `scripts/`（非 archive） |

## 使用约定

1. **不要把它们当作可用接口或架构依据**；需要能力请用现役入口：
   测试 → `pytest tests/`；评测 → `python -m junqi benchmark` / `python -m junqi gate`；
   数据集 → `python -m junqi export_dataset`。
2. **不要在这些脚本上继续加功能**。若某个结论仍需长期守门，把它写成
   `tests/` 下的正式用例；若仍需长期复跑，把它提到 `scripts/`（非 archive）并补测试。
3. 这些脚本**不在** pytest 收集范围内（`pytest.ini` 限定 `testpaths = tests`，
   文件名也不匹配 `test_*.py`），因此它们的失败不会体现在测试结果里——
   这也正是它们不能作为质量证据的原因。
4. 若 `junqi/train_rl.py` 里的重复统计原语（`wilson_lower_bound` / `decide_promotion`）
   将来被删除，本目录脚本会 ImportError。届时请按第 1 条改用 `eval_gate` 的等价函数，
   或直接删除这些脚本，**不要**为了兼容它们而保留死代码。
