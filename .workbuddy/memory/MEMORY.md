# junqi_engine 项目长期记忆（跨会话约定）

## 环境与启动（最容易踩坑）

- **虚拟环境不在工程内**：位于同级 `E:\Local code\军棋\venv_junqi_engine`（2026-09-15 移出，
  原 `junqi_engine/venv/`）。Python 3.11.9 + torch 2.5.1+cu121 + pytest 9.1.1。
  `venv/` 在 `.gitignore` 内，移动不影响版本控制。
- **激活虚拟环境**：`.\activate_env.ps1`（PowerShell）或 `activate_env.cmd`（cmd）——
  根目录脚本自动定位，不必记 `..`。手写路径注意是 **`..\venv_junqi_engine\...`**
  （上一级，不是 `.\`）；激活后脚本会做 `import torch` 依赖自检。
- **一律用工程根 `run.bat <子命令>` 启动**（自动定位解释器，无需先激活）：
  `run.bat test` / `run.bat gui` / `run.bat train_rl --epochs 5 --games 24`。
  等价完整路径：`../venv_junqi_engine/Scripts/python.exe -m junqi <子命令>`。
- **不要用 PATH 上的 `python`**：PATH 上是系统 Python（3.13/3.14，**无 torch/numpy**）。
  用它跑 `python -m junqi ...` 会在包导入阶段失败（`junqi/__init__.py` 经 ai/hybrid_engine
  连带 `import torch`）。现在会给出中文可操作提示，但根本解法是用 run.bat。
- **⚠️ 残留 `(venv)` 激活陷阱**：venv 被移动/删除后，旧会话仍显示 `(venv)`，
  但 PATH 首项指向已失效目录 → `python` 静默落到别的解释器 → 报 `No module named 'torch'`。
  处置：`deactivate` 或**重开终端**，再用 `activate_env.ps1` 激活。
  `activate_env.*` 已内置检测与清理。
- **⚠️ PowerShell 5.1 编码陷阱**：PS 5.1 对**无 BOM** 的 UTF-8 `.ps1` 按本地编码（GBK）读取，
  中文注释会变乱码并抛出"缺少右 }"这类假语法错误。
  → **仓库内任何含非 ASCII 的 `.ps1` 必须带 UTF-8 BOM**（已有测试守卫）。
- **本 bash 环境 shim 已损坏**：`dirname: command not found`、`cd: null directory`；
  且 **shell 里带反斜杠的 Python `-c` 字符串会被吞掉**（会让 `str.count()` 恒为 0）。
  → 含路径的脚本请写成文件再执行；路径用正斜杠或 `chr(92)` 拼接。
  → coreutils（ls/grep/head/tail/wc）不可用，文件/内容检索用 Read/Glob/Grep 工具或 Python。
  → PowerShell 工具的 stdout 不落回上下文，验证 PS 脚本要把输出重定向到文件再读。

## 危险操作禁令（血的教训）

- **删除文件一律用 Python 文件系统操作**（`os.remove` / `shutil.rmtree`），
  **禁止用 `git rm`**：2026-09-15 用 `git rm -r` 时命令被 SIGTERM 中断，遗留
  `.git/index.lock`，随后 `junqi/` 与 `scripts/` 共 96 个文件从工作区消失。
  恢复靠：删 stale lock → `git checkout HEAD -- junqi scripts`（本环境会自动提交，
  所以 HEAD 含最新改动）。核验文件是否还在用关键词检查，不要只信 `git status`。
- **不跑长 `&&` 链的 git 写命令**；git 操作前后确认无 `.git/index.lock`。
- 不用 `dangerouslyDisableSandbox`；不主动删除 `models/`、`军旗复盘/`、`datasets/` 等数据。

## 项目硬约束（来自 AGENTS.md / AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）

- 改代码前先读 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md`，并声明本次修改所属阶段 P0–P4。
- **先加/改测试，再改实现**；完成后报告：改动文件、实验种子、测试结果、未解决风险。
- 禁止：真实暗子身份进公共 Policy；吃子/挖雷等中间奖励；三个独立阶段模型；
  只看 loss 覆盖 `best.pt`；P0 未过就长训练。
- Value 标签严格按官方 `list.cfg` 终局码：1/21/22/23 → ±1；40/42/43 → 0；
  20/24 → 不赋 Value（`dataset.terminal_label_from_meta` 是唯一真源，
  统计口径必须走 `outcome_bucket_from_meta`，禁止各写一遍码表）。
- 数据集默认目录 `datasets/p1_v3`（`dataset.DEFAULT_P1_DIR`），
  低于 3.0.0 的旧数据集会被 `check_p1_version` 拒载。
- 门控唯一真源是 `junqi/eval_gate.py::run_gate`（配对同牌 + Wilson + 三元 SPRT）；
  训练主循环的 `train_rl.evaluate_gate` 只是它的薄封装。

## 架构要点

- 在线传统搜索引擎是 **`junqi/search.py::ExpertSearchEngine`**（`ai.py` / `hybrid_engine.py` 复用）。
  `junqi/expert/` 已于 2026-09-15 **彻底删除**（不可导入、fan-in 0），
  记录见 `docs/06-References/DEPRECATED_EXPERT_PACKAGE.md`。不要重建同名包。
- `junqi/net.py::save()` 写出的是**包装字典** `{"model_state": ...}`，不是裸 `state_dict`。
  `torch.load` 之后必须先 `JunqiNet.unwrap_state_dict()`，否则 `load_state_dict(strict=False)`
  会静默零载入（这正是历史上"best 对手是随机网络"的根因）。
- 热启动**禁止** `net = JunqiNet.load_from_file(...)` 重绑变量（会让 optimizer 脱钩、
  权重零更新）；用 `train_rl.warmstart_candidate(net, optimizer, path, ...)`。
- `StratifiedReplayBuffer` 样本契约：policy `(state, mask, target, phase)`、
  value `(state, z_cls, is_world)`，索引写死。
- `scripts/archive/` 与 `tests/utils/` 是**归档脚本**（见各自 README），
  不是 API、不进 pytest 收集范围；不要在其上继续开发。
- `junqi/benchmark.py::create_benchmark_suite`、`run_training` 等 21 个函数 >120 行，
  属已知结构性债务；拆分需单独评估，不要顺手改。
