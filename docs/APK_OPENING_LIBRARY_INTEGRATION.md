# APK 开局库集成到 P1 ExpertSearchEngine

**日期**: 2026-09-16  
**阶段**: P1 (传统搜索增强)  
**状态**: ✅ 代码已集成；⚠ **未做对局级验证**（见"口径与量纲""实测效果"两节）

## 概述

把 `junqi/apk_engine.py::eval_apk_flip_root` 里**既有**的开局库常数
（`CENTER_CAMP_FLIP_POSITIONS` / `+150`，与 `src_cpp/src/eval_apk.cpp` 同一出处）
移植到 `ExpertSearchEngine._score_action` 的翻棋排序中，使专家引擎在纯开局阶段
也优先翻中心行营旁的黄金格。**新增的是"用在哪里"，不是常数本身。**

## 口径与量纲（重要，不要照着 APK 的数字去理解本集成）

| | APK 原生（`eval_apk_flip_root`） | 本引擎（`ExpertSearchEngine._score_action`） |
|---|---|---|
| 分数量级 | `SI=2560` / `连长=40` / 行营位 40~50 | `safety_score` 20 000、`camp_expansion` 最高 120 000、`territory_bias` ±25 000 |
| `+150` 的相对权重 | 约 4 个连长 ⇒ **强偏好** | **0.75%** ⇒ 只相当于**同分并列时的 tie-break** |
| 实测效果 | — | 纯开局首翻 8/8 落在黄金格（改前 8/8 落非黄金格） |

⇒ 本移植**不是**把 APK 的偏好强度照搬，而是**复用了它的格位选择**。

**副作用（需知情）**：6 个黄金格在本引擎里同分（同处 `territory_bias=25000` 档），
tie 由候选生成序决定 ⇒ 首翻**恒为 (3,1)**，开局被确定化。
要保持多样性需给格位分档或引入受控随机 tie-break，**属待决策项，本轮未改行为**（2026-09-16 第十六批）。

## 技术实现

### 1. 常量定义 (`junqi/search.py:35-42`)

```python
# APK 开局库增强（2026-09-16）
# 官方 libjunqi.so 的中心行营黄金翻棋格 (对齐 0x5a3c0)
APK_CENTER_CAMP_FLIP_POSITIONS: Set[Tuple[int, int]] = frozenset({
    (3, 1), (3, 3), (4, 2),   # 上半场中心行营辐射位
    (7, 2), (8, 1), (8, 3),   # 下半场中心行营辐射位
})

# APK 开局优先翻棋加分（纯开局无子时生效）
APK_FLIP_ROOT_BONUS = 150.0
```

### 2. 评分逻辑集成 (`junqi/search.py:397-409`)

在 `_score_action()` 方法的翻棋分支中添加 APK 开局偏好：

```python
elif act.kind == "flip":
    pos = act.frm
    r, c = pos
    safety_score = 0.0
    
    # APK 开局库增强：优先翻中心行营周围的黄金暗子格
    apk_flip_bonus = 0.0
    revealed_friendly_pieces = sum(1 for p in state.board.values() 
                                  if p.revealed and p.color == my)
    if revealed_friendly_pieces == 0 and pos in APK_CENTER_CAMP_FLIP_POSITIONS:
        apk_flip_bonus = APK_FLIP_ROOT_BONUS
```

**触发条件**:
- **阶段检查**: `revealed_friendly_pieces == 0`（全盘**无己方明子**；注意不是"仅第一手"，
  在翻出己方明子之前一直有效）
- **位置检查**: `pos in APK_CENTER_CAMP_FLIP_POSITIONS`（黄金翻棋格）
- **奖励值**: `+150.0`。⚠ 这是**相对本引擎量纲的 tie-break**（同档安全分 20 000、
  领地分 25 000 ⇒ 占 0.33%），**不是**"高于普通翻棋基础分"的强偏好；
  详见"口径与量纲"一节

### 3. 与现有优先级融合

APK 开局库与其他专家评分项协同工作：

```python
# 最终翻棋分数 = 安全评分 + 行营扩展奖 + 领地偏好 + APK 开局加成
total_score = safety_score + camp_expansion_bonus + territory_bias + apk_flip_bonus
```

## 设计原理

### 为什么选择这些位置？

APK 选择的 6 个黄金格都是**中心 4 个行营的紧邻格**：

```
      铁路网络示意
    0  1  2  3  4
  ┌─────────────┐
3 │  ★    ★     │ ← 上半区黄金格 (3,1), (4,2), (3,3)
  │             │
7 │  ★    ★     │ ← 下半区黄金格 (7,2), (8,1), (8,3)
  │  ★    ★     │
  └─────────────┘
```

**优势**:
1. **多向辐射**: 每个黄金格可快速到达 2-3 个行营
2. **对称结构**: 上下半场镜像对称，符合公平性原则
3. **战术枢纽**: 控制中心线，便于大子机动和敌情侦察

### 为什么只在纯开局生效？

```python
if revealed_friendly_pieces == 0 and pos in APK_CENTER_CAMP_FLIP_POSITIONS:
    apk_flip_bonus = APK_FLIP_ROOT_BONUS
```

**理由**:
1. **动态评估**: 一旦有己方明子暴露，局面复杂度↑，单纯位置价值↓
2. **避免僵化**: 后续决策应基于具体子力对比而非固定偏好
3. **经验验证**: APK 原版也是仅在首翻时使用此启发式

## 实测效果（**没有**"预期提升"百分比）

⚠ 本节此前给出过 `+40% / +25% / +60%` 的"性能提升"，**没有任何实测依据，已删除**。
在拿到对局级 A/B 之前，只能声明以下**可复现**的事实：

| 事实 | 证据 |
|---|---|
| 纯开局首翻恒定落在黄金格 | `scratch/probe_review_apk_first_flip.py` → 8/8 命中黄金格（改动前 8/8 落非黄金格） |
| 但 6 格同分，实际恒为 `(3,1)` | 同一探针：8 个不同 deal 全部选 `(3,1)` |
| 对棋力 / 据点率 / 首翻效率的影响 | **未知** —— 需对局级 A/B（见下） |

想做对局级验证时的正确口径：`gate --init-set eval_sets/endgame.jsonl`（随机发牌开局
100% 循环判和，无分辨力），并先跑**同模型空测**确认得分率恰为 0.5000。

### 实战行为变化

**Before**: 首翻落非黄金格（本引擎在 `territory_bias` 同档内由候选生成序决定）
```
原始布局（全盘 50 枚暗子）：
首选翻棋：(2, 2) 等非黄金格（分数 45000.0）
```

**After**: 首翻落黄金格
```
首选翻棋：黄金格（分数 45150.0 = 45000 + APK 150）✓
但 6 个黄金格同分 ⇒ 实测恒为候选生成序第一的 (3, 1)
```

## 与 APK 引擎的对齐度

### 常数与门槛一致的部分

| 特性 | APK 原版 | P1 ExpertSearch | 对齐状态 |
|------|---------|-----------------|---------|
| 黄金格坐标 | `{(3,1),(3,3),(4,2),(7,2),(8,1),(8,3)}` | 同上 | ✅ 常数同源（已核对） |
| bonus 值 | `+150.0`（尺度 ~2560） | `+150.0`（尺度 ~20 000~125 000） | ⚠ 数值相同、**语义不同**（强偏好 vs tie-break） |
| 触发条件 | 无己方明子时 | 无己方明子时 | ✅ 一致 |
| 作用范围 | 根节点翻棋动作 | 翻棋动作排序（根节点为主；Python 路径下内层也可命中） | ⚠ 不完全相同 |

### 额外增强部分

P1 ExpertSearch 相比 APK 原版增加了：

- ✅ **多层专家评分**: APK 只有简单物质分，P1 有动态制霸、死区势能等
- ✅ **Star1 几率剪枝**: APK 不使用概率树，P1 精确处理翻棋期望
- ✅ **QSearch 深度**: P1 默认 16 层吃子链 vs APK 的 8/12/16 档
- ✅ **迭代加深时间管理**: P1 更灵活的预算控制 vs APK 的 25% 硬截断

## 使用示例

```python
from junqi.state import GameState
from junqi.config import RuleConfig
from junqi.search import ExpertSearchEngine

# 初始化
cfg = RuleConfig()
state = GameState.initialize(cfg)
engine = ExpertSearchEngine()

# 获取所有合法动作
acts = state.legal_actions()
flips = [a for a in acts if a.kind == "flip"]

# 计算翻棋动作评分
for act in flips[:6]:  # 只取前 6 个黄金格
    score = engine._score_action(act, state, ply_depth=0)
    is_golden = act.frm in {(3,1), (3,3), (4,2), (7,2), (8,1), (8,3)}
    print(f"Pos {act.frm}: score={score:.1f}, golden={is_golden}")

# 输出示例（首次翻棋、全盘皆暗时的真实分值）：
# Pos (3, 1): score=45150.0, golden=True    ← 20000(safety) + 25000(territory) + 150(APK)
# Pos (3, 3): score=45150.0, golden=True    ← 6 个黄金格同分 ⇒ tie 由候选生成序决定
# Pos (2, 2): score=45000.0, golden=False   ← 同一档次但无 bonus
# Pos (5, 0): score=40000.0, golden=False   ← territory_bias 降档（r=5/6）
# Pos (0, 0): score=0.0,     golden=False   ← 底线，territory_bias = −20000
#
# ⚠ 注意量纲：+150 相对 45000 只占 0.33%，因此它**只在同分并列时起作用**。
#   若把常数放大到与 safety/territory 同量级，会反过来压过营地逻辑 ⇒ 属待决策项。
```

## 验证现状（哪些做了、哪些没做）

### 已做

- **常数出处核对**：`CENTER_CAMP_FLIP_POSITIONS` 与 `+150` 在
  `junqi/apk_engine.py`（92-96、206-209）、`src_cpp/src/eval_apk.cpp`（116-125）
  中已存在且一致；APK 侧行为由 `tests/test_p4_pure_apk_alignment.py` 覆盖。
- **集成后的行为**：`scratch/probe_review_apk_first_flip.py` 实测首翻 8/8 落黄金格
  （改动前 8/8 落非黄金格）。
- **等价性**：本改动的加成只出现在 Python 的 `_score_action`（根节点排序），
  不进入 C++ 子树；`perf_baseline verify --cpp` 通过。

### 未做（**不要误以为已结论**）

1. **对局级 A/B**：启用 vs 禁用该加成，用有分辨力的靶场比较。
   ```bash
   python -m junqi gate --a expert2 --b expert2 --seeds 30 \
       --init-set eval_sets/endgame.jsonl --out reports --out-name gate_apk_vs_noapk
   ```
   （该加成只在"全盘无己方明子"时生效，随机发牌开局下 `hybrid2` 自对局
   100% 循环判和、无分辨力；务必先跑同模型空测确认 0.5000。）
2. **非开局阶段确实禁用**：需要一条把"己方已有明子"局面下的 bonus 归零钉死的测试
   （目前 `tests/test_p1_advanced_enhancements.py` 未覆盖这一项）。
3. **首翻确定化**（恒 `(3,1)`）是否需要打散。

## 相关资源

- **常数出处**：`junqi/apk_engine.py`（`CENTER_CAMP_FLIP_POSITIONS` 第 92-96 行、
  `eval_apk_flip_root` 第 206-209 行）与 `src_cpp/src/eval_apk.cpp`（116-125 行）；
  APK 对齐测试 `tests/test_p4_pure_apk_alignment.py`。
- **反汇编取证脚本**（不是产物，勿当源码引用）：`scratch/disasm_ai_sub.py`、
  `scratch/disasm_bc.py`、`scratch/disasm_ai_full.py` —— 其中 `0x5a3c0` 被标注为
  `move_score`、`0x5a678` 为 `quiescence`。
  ⚠ 本文此前引用的 `apk_extracted/ai_0x5a3c0.asm`、`test_simple_apk.py`、
  `scripts/benchmark_apk_vs_expert.py` **在仓库中不存在**，已更正。
- APK 引擎复刻：`junqi/apk_engine.py`（含 `APK_LEVEL_SPECS`）。
- 专家排序：`junqi/search.py::_score_action()`（MVV-LVA + TT + 杀手/历史启发）。
- 首翻行为的可复现探针：`scratch/probe_review_apk_first_flip.py`。

## 结论

✅ **APK 开局库常数已集成进 P1 ExpertSearchEngine 的翻棋排序**

- 复用了官方 APK 的 6 个黄金翻棋格选择（常数与 APK 侧同源、已核对）；
- 在"全盘无己方明子"阶段给予 `+150`，**在本引擎里是 tie-break 而非强偏好**（见量纲一节）；
- 与既有专家排序不冲突（累加在 `safety + camp_expansion + territory` 之后）；
- **对棋力无已验证收益**；首翻被确定化为 `(3,1)`，是否可接受需决策。

---

**维护者**: Hu  
**下次审查**: 拿到对局级 A/B（或决定首翻是否需要打散）之后
**相关文件**: `junqi/search.py`, `junqi/apk_engine.py`, `junqi/ai.py`
