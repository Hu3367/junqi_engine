# 🎯 军棋翻棋棋理体系 v2.0

> **版本**: 2.0  
> **日期**: 2026-09-05  
> **目标**: 建立完整的"不完全信息博弈"棋理本体  
> **适用范围**: AI 训练、策略设计、人机对战

---

## 📋 目录

1. [不完全信息推理体系](#第 1 章不完全信息推理体系)
2. [空间与势力控制](#第 2 章空间与势力控制)
3. [威胁体系](#第 3 章威胁体系)
4. [Tempo 经济学](#第 4 章 tempo 经济学)
5. [条件子力价值](#第 5 章条件子力价值)
6. [战略计划层](#第 6 章战略计划层)
7. [残局定式库](#第 7 章残局定式库)

---

## 第 1 章 不完全信息推理体系 ⭐⭐⭐⭐⭐

### 1.1 为什么不完全信息是核心？

#### ❌ 传统评估的错误假设

大多数棋牌游戏的 AI 使用这种简化的价值计算：

```python
def naive_material_value(state):
    my_value = sum(piece.value for piece in state.my_pieces if revealed)
    enemy_value = sum(piece.value for piece in state.enemy_pieces if revealed)
    hidden_expectation = average_piece_value * hidden_count
    return (my_value + hidden_expectation) - (enemy_value + hidden_expectation)
```

**致命缺陷**: 将"暗子"视为等概率的"期望棋子"。

#### ✅ 正确理解：位置决定战略意义

两个暗子，即使概率分布完全相同，战略意义也完全不同：

| 特征 | 暗子 A（敌方军旗旁） | 暗子 B（边缘角落） |
|------|-------------------|------------------|
| 概率分布 | 80% 小子，20% 大子 | 80% 小子，20% 大子 |
| **实际意义** | **可能是雷或炸弹保护军旗** | **只是普通的边缘防守** |
| 风险等级 | 🔴 高危 | 🟢 低危 |
| 应对优先级 | 立即侦察 | 可延后处理 |

**结论**: 不能只用"剩余池平均价值"来评估暗子。

---

### 1.2 贝叶斯信念建模

#### 每个暗子都是一个独立的不确定事件

对于每个未翻开的暗子 `pos`，我们需要维护：

```python
class HiddenPieceBelief:
    def __init__(self, pos, initial_pool):
        """
        初始化时，每个暗子的概率 = 初始配对的均匀分布
        
        Example:
        self.belief[pos] = {
            'SI': 1/25,  # 司令有 1 个在池中
            'JUN': 1/25,  # 军长有 1 个
            ...
            'LEI': 3/25   # 地雷有 3 个
        }
        """
        pass
    
    def update_with_observation(self, observation_type, data):
        """
        当发生某个观察事件时，更新所有暗子的概率分布
        
        观察类型:
        - "piece_revealed": 某位置翻出具体棋子
        - "piece_killed": 某棋子被吃掉
        - "piece_moved": 己方明子移动（可能暗示前方无大子）
        - "flip_avoidance": 对手长期回避某个区域（可能是雷阵）
        """
        pass
    
    def get_confidence_score(self) -> float:
        """
        返回当前信念的"确定性评分"
        0 = 完全未知
        1 = 几乎确定
        
        用于判断是否应该主动侦察该暗子
        """
        pass
```

#### 实战示例

**初始状态**: 暗子池中有 2 个司令

```json
{
  "hidden_pos_1": {
    "SI": 2/50,
    "JUN": 1/50,
    "SHI": 2/50,
    ...
    "LEI": 3/50
  },
  "hidden_pos_2": {
    "SI": 2/50,  // 同样的初始概率
    "JUN": 1/50,
    ...
  }
}
```

**观察 1**: 红方在第 20 回合翻出了唯一的司令

```json
{
  "hidden_pos_1": {
    "SI": 0,      // ← 更新！现在知道这里不可能是司令
    "JUN": 2/49,  // ← 概率重新归一化
    ...
  },
  "hidden_pos_2": {
    "SI": 0,
    "JUN": 2/49,
    ...
  }
}
```

**观察 2**: 黑方连续 10 回合避开了右侧区域

```json
{
  "hidden_pos_right_side": {
    "SI": 0,
    "JUN": 0,
    "SHI": 0,
    "LEI": 8/40,  // ← 怀疑是雷阵，地雷概率大幅提升
    "ZHA": 5/40   // 也可能是炸弹防守
  }
}
```

**关键洞察**: 对手的**行为模式本身也是信息源**。

---

### 1.3 信息价值量化

#### 定义：Reveal Value

翻出某个暗子的"信息增益"可以这样衡量：

```python
def calculate_reveal_value(pos, belief_system, game_state):
    """
    信息价值 = 翻前不确定性 - 翻后不确定性
    
    考虑因素:
    
    1. 剩余池变化
       - 如果翻出的是稀有大子（如司令），会大幅降低其他位置的概率
       - 如果翻出的是普通小棋子，信息价值较低
    
    2. 条件概率调整
       - 某些位置的暗子概率会同时调整（因为牌池共享）
       - 调整幅度越大，信息价值越高
    
    3. 战略影响
       - 如果这个暗子是保护军旗的关键，信息价值极高
       - 如果是边路无关紧要的暗子，价值低
    
    4. 机会成本
       - 翻这个暗子需要消耗 1 步
       - 但可能会给对手 2 步的行动权
    """
    
    before_entropy = calculate_entropy(belief_system.get_all_beliefs())
    
    # 模拟翻出后的状态
    simulated_belief_after_flip = simulate_flip(belief_system, pos)
    after_entropy = calculate_entropy(simulated_belief_after_flip)
    
    information_gain = before_entropy - after_entropy
    
    strategic_multiplier = calculate_strategic_importance(pos, game_state)
    
    return information_gain * strategic_multiplier
```

#### 熵（Entropy）的计算

```python
import math

def calculate_entropy(probability_distribution):
    """
    计算概率分布的熵值
    
    H(P) = -Σ p(x) * log(p(x))
    
    熵越大 → 越不确定
    熵越小 → 越确定
    """
    entropy = 0.0
    for prob in probability_distribution.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy
```

**示例**:
- 初始状态：熵 ≈ 4.3 bits（高度不确定）
- 发现一方司令已出：熵 ≈ 3.8 bits（部分确定）
- 全部翻开：熵 = 0 bits（完全确定）

---

### 1.4 排除法推理系统

#### 基于已知信息的逻辑推断

```python
class EliminationReasoner:
    def __init__(self):
        self.known_pieces = {
            'r': [],  # 红方已知的棋子列表
            'b': []   # 黑方已知的棋子列表
        }
        self.pool_composition = {
            'r': ['SI']*1 + ['JUN']*1 + ['SHI']*2 + ...,
            'b': [...]  # 双方初始配置相同
        }
    
    def update_known(self, color, rank, action='revealed'):
        """
        当某个棋子被揭示或被吃掉时，更新知识库
        
        action: 'revealed' | 'killed'
        """
        self.known_pieces[color].append((rank, action))
    
    def get_probability_for_position(self, pos, color=None):
        """
        计算某个暗子位置属于某个棋子的概率
        
        P(piece = R at pos | known_info)
        
        计算方法:
        1. 初始概率 = pool[R] / total_hidden
        2. 排除已知出现的棋子
        3. 归一化剩余概率
        """
        remaining_pool = self.get_remaining_pool()
        total_remaining = sum(remaining_pool.values())
        
        probabilities = {}
        for rank, count in remaining_pool.items():
            probabilities[rank] = count / total_remaining
            
        return probabilities
    
    def detect_impossible_pieces(self, pos):
        """
        识别某个暗子位置"不可能是什么棋子"
        
        例如:
        - 敌方司令已经出现 → 所有敌方暗子都不是司令
        - 敌方只剩工兵未出 → 某个暗子一定是工兵
        """
        impossible = {}
        
        # 检查哪些棋子已经完全暴露
        completely_exposed = self.find_exhausted_ranks()
        
        for rank in completely_exposed:
            impossible[rank] = True
            
        return list(impossible.keys())
```

#### 实战案例

**场景**: 开局第 15 回合

**已知信息**:
- 红方已翻出：司令×1、军长×1、师长×2、炸弹×1、工兵×3、地雷×3
- 黑方已翻出：司令×1、军长×1、师长×1、旅长×1...

**推理结果**:
```
暗子 A 的位置: 
  ❌ 不可能是司令 (红方司令已出)
  ❌ 不可能是军长 (黑方军长已出)
  ✅ 最有可能是师长 (还剩 1 个在池)
  ✅ 有可能是旅长 (还剩 1 个在池)

暗子 B 的位置:
  ❌ 不可能是工兵 (双方工兵都已全出)
  ❌ 不可能是地雷 (红方地雷已全出)
  ✅ 最有可能是师长/旅长/团长
```

**战略决策**:
- 对暗子 A：谨慎接近（可能是师长）
- 对暗子 B：相对安全（不是工兵/地雷）

---

### 1.5 基于行为的反向推理

#### 对手的行为揭示了什么信息？

##### 行为模式 1：避开某个区域

```
观察到：黑方连续 10 回合没有攻击右侧中路
推理：
- 这个区域可能有雷阵
- 或者有大子在埋伏
- 或者工兵已经被清除，无法突破
```

##### 行为模式 2：频繁翻棋

```
观察到：红方连续翻了 5 个暗子
推理：
- 可能在收集信息
- 或者试图快速激活某个大子
- 也可能是在浪费 tempo（糟糕的策略）
```

##### 行为模式 3：优先占领行营

```
观察到：黑方总是抢占中央行营
推理：
- 可能在构筑防线
- 或者为后续进攻做准备
- 或者避免被我方的铁路切入
```

---

### 1.6 信息优势的量化工具

#### Information Advantage Score

```python
def calculate_information_advantage(game_state, player_color):
    """
    计算双方在信息层面的优劣
    
    因素:
    1. 已知棋子数量差
       我方揭示的敌方棋子数 - 对方揭示的我方棋子数
    
    2. 位置信息清晰度
       我对某个区域的暗子了解程度 - 对方对我的了解程度
    
    3. 推测准确率
       我的推测与实际翻出的匹配度
    
    4. 信息时效性
       最近 5 回合内获得的新信息量
    """
    
    factors = {
        'pieces_revealed_diff': my_revealed_enemy_pieces - enemy_revealed_my_pieces,
        'location_clarity': calculate_clarity_score(game_state),
        'prediction_accuracy': track_prediction_success_rate(),
        'freshness': measure_recent_information_flow()
    }
    
    # 加权平均
    weights = {'pieces_revealed_diff': 0.4, 'location_clarity': 0.3,
               'prediction_accuracy': 0.2, 'freshness': 0.1}
    
    score = sum(factors[k] * weights[k] for k in factors)
    
    # 归一化到 [-1, 1]
    normalized = normalize_to_minus_1_1(score)
    
    return normalized
```

**解读**:
- +0.8: 我有巨大的信息优势
- +0.3: 略优于对方
- 0.0: 信息对称
- -0.3: 处于信息劣势
- -0.8: 非常被动，不知道敌情

---

### 1.7 侦察策略选择

#### 什么时候应该翻棋侦察？

```python
def should_reveal_position(pos, belief_system, current_tempo, options):
    """
    决策函数：是否要翻某个暗子
    
    权衡因素:
    
    ✅ 支持翻棋的理由:
    1. 该位置的战略价值极高（如靠近军旗）
    2. 信息不确定性过高（entropy > threshold）
    3. 当前 tempo 充裕，不怕失去先手
    4. 推测这里有重要棋子（如可能的司令位置）
    
    ❌ 反对翻棋的理由:
    1. 位置价值低（边缘角落）
    2. 概率已经很清晰（排除法已做）
    3. 当前 tempo 紧张（需要抢关键点）
    4. 翻出后会暴露战术意图
    """
    
    strategic_importance = evaluate_position_importance(pos, game_state)
    uncertainty_level = belief_system.calculate_entropy_for_position(pos)
    tempo_pressure = evaluate_tempo_pressure(current_turn)
    
    if strategic_importance > 0.7 and uncertainty_level > 0.5:
        return True, f"高战略价值 + 高度不确定 = 立即侦察"
    
    if tempo_pressure > 0.8:
        return False, "tempo 压力大，先别翻棋"
    
    if uncertainty_level < 0.2:
        return False, "已经很清楚了，不需要侦察"
    
    return random.choice([True, False]), "模糊情况，随机决策"
```

---

### 1.8 实践要点总结

✅ **核心原则**:
1. 每个暗子都是独立的概率分布，不是"平均棋子"
2. 位置比概率更重要（军旗旁的暗子 ≠ 边角的暗子）
3. 对手的行为本身就是强大的信息源
4. 信息价值 ≠ 棋子价值（有时翻个小棋子价值巨大）
5. 侦察时机取决于 tempo 压力和战略需求

❌ **常见错误**:
1. 把所有暗子都当成同等价值
2. 忽视对手行为中的信息
3. 盲目翻棋而不考虑 tempo 损失
4. 认为"知道了暗子身份"就万事大吉

🎯 **正确的思维框架**:
```
翻棋 = 获取信息
     = 支付 tempo 成本
     = 揭示潜在威胁/机会
     = 调整整个局面的概率分布
     
每一步都要问：
- 翻这个暗子能告诉我什么？
- 知道了之后我能做什么更好的决策？
- 现在翻 vs 稍后翻有什么区别？
- 不翻会有什么问题？
```

---

## 第 2 章 空间与势力控制 ⭐⭐⭐⭐⭐

### 2.1 为什么空间比棋子更重要？

#### ❌ 传统思维的错误

初学者常认为："我有司令、你有排长，所以我赢"

但高手知道："我能控制铁路枢纽，你只能困守行营，所以我赢"

**核心洞察**: 在军棋翻棋中，**空间价值往往超过子力价值**。

#### 空间的三层含义

| 层次 | 定义 | 战略意义 |
|------|------|---------|
| **移动空间** | 棋子能到达的格数 | 机动性、灵活性 |
| **控制空间** | 能有效威胁的区域 | 影响力范围 |
| **结构空间** | 关键节点和路径 | 战略性通道 |

---

### 2.2 Mobility 的多层概念

#### 定义：Mobility-k

**Mobility-1**: 一回合内可达区域

```python
def mobility_1(board, piece_pos):
    """
    计算从某个位置出发，一回合能到达的所有格子
    
    包括:
    - 相邻格子 (公路移动)
    - 铁路直线路径 (滑行)
    - 工兵飞行路径 (如果适用)
    """
    reachable = set()
    
    # 公路移动：上、下、左、右
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = piece_pos[0] + dr, piece_pos[1] + dc
        if is_valid_move(piece_pos, (nr, nc)):
            reachable.add((nr, nc))
    
    # 铁路滑行
    if is_rail(piece_pos):
        for direction in HORIZONTAL_VERTICAL:
            slide_positions = rail_slides_from(piece_pos, direction)
            reachable.update(slide_positions)
    
    return reachable
```

**Mobility-2**: 两回合内可达区域（考虑中间步骤）

```python
def mobility_2(board, piece_pos):
    """
    计算两回合能到达的区域
    
    关键点:
    1. 第一回合可能经过的行营/铁路
    2. 第二回合从这些中转点延伸
    3. 考虑敌方阻挡的可能性
    
    示例:
    初始位置：(5,0)
    M1: (4,0), (6,0), (5,1)
    M2: (3,0), (7,0), (5,2), (4,1), (6,1), ...
    """
    m1 = mobility_1(board, piece_pos)
    m2 = set()
    
    for intermediate_pos in m1:
        m2_intermediate = mobility_1(board, intermediate_pos)
        m2.update(m2_intermediate)
    
    return m2
```

**Mobility-3**: 三回合潜在可达区域（加入推测成分）

```python
def mobility_3(board, piece_pos, belief_system):
    """
    计算三回合可能达到的区域
    
    考虑:
    1. 最乐观的移动路线（假设无阻挡）
    2. 基于信念的预测（某些暗子可能是障碍）
    3. 最佳路径规划（优先选择铁路/行营）
    
    这是 AI 评估"长远潜力"的基础
    """
    m2 = mobility_2(board, piece_pos)
    m3 = set()
    
    for pos in m2:
        # 对于未知区域，根据概率加权
        expected_reach = weighted_mobility_1(pos, belief_system)
        m3.update(expected_reach)
    
    return m3
```

#### 实战对比：不同 Mobility 的价值

**情境**: 红方司令 vs 黑方司令

| 参数 | 红方司令 | 黑方司令 | 分析 |
|------|---------|---------|------|
| 位置 | (4,2) 中央行营 | (10,4) 角落 | - |
| M1 | 8 格 | 4 格 | 红方优势 |
| M2 | 23 格 | 12 格 | 红方扩大优势 |
| M3 | 47 格 | 21 格 | 红方完全控制中场 |
| **实际价值** | **高机动大子** | **被困大子** | - |

**结论**: 两个都是司令，但红方司令价值可能是黑方的**2 倍**以上。

---

### 2.3 战略节点识别

#### 哪些格子最重要？

并非所有格子都有同等价值。我们需要系统地识别**战略节点**。

##### 节点类型分类

```python
STRATEGIC_NODES = {
    'central_rail_junctions': {  # 中央铁路交汇点
        'positions': [(5,0), (5,4), (6,0), (6,4)],
        'value': 0.95,
        'reason': '连接南北的快速通道',
        'control_priority': 'MAX'
    },
    
    'frontline_crossings': {  # 前线穿越点
        'positions': [(5,2), (6,2)],
        'value': 0.85,
        'reason': '唯一可直线穿越前线的公路',
        'control_priority': 'HIGH'
    },
    
    'camp_cluster_centers': {  # 行营集群中心
        'positions': [(3,2), (8,2)],
        'value': 0.80,
        'reason': '前后方调度的枢纽',
        'control_priority': 'HIGH'
    },
    
    'flag_approach_paths': {  # 旗区接近路径
        'positions': [(1,0), (1,4), (10,0), (10,4)],
        'value': 0.90,
        'reason': '工兵最后冲锋的关键路径',
        'control_priority': 'MAX'
    },
    
    'railway_gates': {  # 铁路闸门
        'positions': [(1,0), (1,4), (10,0), (10,4)],
        'value': 0.88,
        'reason': '控制一侧铁路线的入口',
        'control_priority': 'HIGH'
    }
}
```

##### 节点价值评分算法

```python
def calculate_node_strategic_value(node_type, node_position, game_state):
    """
    计算某个战略节点的实时价值
    
    动态因素:
    1. 控制权归属（己方占了多少？）
    2. 争夺激烈程度（双方距离多远？）
    3. 时机敏感性（当前阶段是否紧急？）
    4. 替代路径可用性（失去这个节点还有备选吗？）
    """
    
    base_value = STRATEGIC_NODES[node_type]['value']
    
    control_factor = evaluate_control_balance(node_position, game_state)
    
    phase_multiplier = get_phase_sensitivity(game_state.phase)
    
    urgency_score = calculate_distance_to_vitality(node_position, game_state)
    
    final_value = base_value * control_factor * phase_multiplier * (1 + urgency_score)
    
    return min(1.0, final_value)  # 归一化
```

#### 实战应用

**开局阶段**:
```
优先级排序:
1. 抢占中央铁路交汇点 (5,0)/(5,4) - 开通南北通道
2. 侦察前线穿越点 (5,2) - 打开公路突破口
3. 争夺中央行营 (3,2)/(8,2) - 建立调度枢纽
```

**中盘阶段**:
```
优先级调整:
1. 封锁敌方旗区接近路径 - 阻止对方工兵
2. 控制敌方铁路线入口 - 限制其大子机动
3. 切断行营集群联系 - 孤立敌方防守
```

**尾盘阶段**:
```
优先级调整:
1. 夺取最后可用的突破点 - 直接攻击雷阵
2. 占领逃生通道 - 确保工兵撤退路线
3. 封锁敌方反扑路径 - 防止被逆袭
```

---

### 2.4 空间压缩与扩张

#### 空间优势的量化指标

```python
def calculate_space_advantage(my_color, enemy_color, board):
    """
    计算双方在空间层面的优劣
    
    三个维度:
    
    1. 活动空间
       我方棋子可达区域总和 / 敌方棋子可达区域总和
    
    2. 控制密度
       我方控制的战略节点数量 / 敌方控制的战略节点数量
    
    3. 路径连通性
       我方各子之间的连接效率 vs 敌方的连接效率
    """
    
    my_mobility_sum = sum(mobility_3(board, pos) 
                         for pos in my_pieces if revealed)
    enemy_mobility_sum = sum(mobility_3(board, pos) 
                            for pos in enemy_pieces if revealed)
    
    mobility_ratio = len(my_mobility_sum) / max(len(enemy_mobility_sum), 1)
    
    # 控制密度
    my_nodes_controlled = count_strategic_nodes_owned(my_color, board)
    enemy_nodes_controlled = count_strategic_nodes_owned(enemy_color, board)
    
    control_ratio = my_nodes_controlled / max(enemy_nodes_controlled, 1)
    
    # 路径连通性（简化版）
    my_connectivity = graph_connectivity_score(my_pieces, board)
    enemy_connectivity = graph_connectivity_score(enemy_pieces, board)
    
    connectivity_ratio = my_connectivity / max(enemy_connectivity, 1)
    
    # 综合评分
    space_advantage = (
        normalize(mobility_ratio) * 0.4 +
        normalize(control_ratio) * 0.4 +
        normalize(connectivity_ratio) * 0.2
    )
    
    # 归一化到 [-1, 1]
    normalized_score = (space_advantage - 1) / 1
    
    return normalized_score
```

#### 解读空间优势评分

| 分数 | 解读 | 策略建议 |
|------|------|---------|
| +0.8 ~ +1.0 | 压倒性空间优势 | 可以主动进攻，扩大战果 |
| +0.3 ~ +0.7 | 适度空间优势 | 稳步推进，逐步挤压 |
| -0.3 ~ +0.2 | 空间基本均势 | 谨慎行事，等待机会 |
| -0.7 ~ -0.3 | 空间劣势 | 收缩防守，寻找反击点 |
| -1.0 ~ -0.8 | 严重空间劣势 | 必须突围或求和 |

---

### 2.5 图论视角下的棋盘

#### 将棋盘视为图

```
顶点集 V: 所有非行营格子
边集 E: 合法移动关系（公路邻接 + 铁路连接）

G = (V, E)

每个位置就是一个节点，每个合法移动就是一条边。
```

##### 关键图论概念的应用

###### 1. 最短路径

```python
from collections import deque

def shortest_path(board, start, end, avoid_enemies=False):
    """
    BFS 计算从一个点到另一个点的最短步数
    
    如果有敌人阻挡，需要绕行
    """
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        if current == end:
            return path
        
        for neighbor in neighbors(current, board):
            if neighbor not in visited and not_blocked(neighbor, avoid_enemies):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None  # 无法到达
```

**用途**:
- 评估工兵能否及时挖到地雷
- 判断大子能否快速支援防线
- 计算突破所需的最小步数

###### 2. 桥节点（Bridges）

```python
def identify_bridges(graph):
    """
    找出图中的桥节点（移除后会导致图不连通的关键点）
    
    例如:
    - 唯一通往后方基地的铁路入口
    - 阻断左右两侧联系的必经之路
    - 工兵突破雷阵的唯一可行路径
    """
    bridges = []
    
    for edge in graph.edges():
        temp_graph = graph.copy()
        temp_graph.remove_edge(edge)
        
        if not is_connected(temp_graph):
            bridges.append(edge)
    
    return bridges
```

**实战价值**:
- 发现敌方防线的脆弱点
- 保护己方的关键通道
- 设计"围而不攻"的封锁战术

###### 3. 割点（Cut Vertices）

```python
def find_cut_vertices(graph):
    """
    找出割点——删除后会增加连通分量数量的节点
    
    这类节点通常是:
    - 单一线路上的关键点
    - 交通枢纽的中心位置
    - 行营连接的瓶颈处
    """
    pass
```

**战术应用**:
- 如果我是进攻方，我要夺取割点
- 如果我是防守方，我要死守割点
- 如果割点在敌手，我需要考虑绕道

###### 4. 支配集（Dominating Set）

```python
def find_minimal_dominating_set(board, strategic_points):
    """
    找到最小的一组点，使得所有其他点都与这组点相邻
    
    应用:
    - 用最少的棋子控制最大的区域
    - 构建高效的前哨网络
    - 设计最优的行营占领方案
    """
    dominating_set = greedy_algorithm(strategic_points)
    
    return dominating_set
```

---

### 2.6 逃生格与生存空间

#### 评估单个棋子的逃生能力

```python
def count_escape_grades(piece_pos, board, enemy_pieces):
    """
    计算某个棋子有多少个安全的逃跑方向
    
    影响因素:
    1. 是否有未被敌方控制的相邻格
    2. 是否有可行的撤退路径
    3. 是否有隐蔽所（行营、大本营）可用
    """
    escape_options = []
    
    for direction in ALL_DIRECTIONS:
        next_pos = piece_pos + direction
        
        # 检查是否是安全格
        if is_safe(next_pos, enemy_pieces, board):
            escape_options.append(next_pos)
    
    return len(escape_options)
```

**战术决策**:
- 逃生格 ≥ 3：相对安全，可以继续行动
- 逃生格 = 1~2：有风险，需谨慎
- 逃生格 = 0：被困！立即寻找解救方案

---

### 2.7 空间转换策略

#### 何时应该主动放弃空间？

```python
def should_trade_space_for_material(position, potential_gain, situation):
    """
    评估是否应该用空间换取物质优势
    
    例如:
    - 牺牲一个行营，换取吃掉敌方司令的机会
    - 让出关键铁路节点，引诱敌方深入陷阱
    
    条件:
    1. 损失的空间不会导致结构性崩溃
    2. 获得的物质足以弥补空间损失
    3. 对手进入新位置后会暴露弱点
    """
    
    space_loss_value = evaluate_lost_area(position)
    material_gain_value = calculate_piece_value(potential_gain)
    positional_weakness = assess_enemy_exposure(new_enemy_position)
    
    # 综合决策
    net_value = material_gain_value + positional_weakness - space_loss_value
    
    if net_value > 0 and not_critical_threat_after():
        return True, f"值得交换：+{net_value:.1f}净收益"
    else:
        return False, "保持原位置更安全"
```

#### 典型案例：弃车保帅

```
局面:
- 红方占据行营 (3,2)，但这里是中路要冲
- 黑方司令在附近潜伏，随时可以吃掉红方
- 红方有两个工兵在准备进攻右侧雷阵

决策:
主动放弃行营 (3,2)，让黑方司令吃子
→ 黑方司令暴露位置
→ 红方其他大子集结反杀
→ 工兵趁机突破雷阵
```

### 2.8 实战开局定式：首翻子力据点化与行营连锁拓荒（基于 1000 局官方实证）⭐⭐⭐⭐⭐

#### 1. 开局全随机规律与“首子即据点”原则
在传统理论中，常有“挑选工兵/排长进驻行营”的理想化假设。但军棋翻棋的底层铁律是：**开局 50 颗暗子完全随机洗乱，无法提前预知翻出的是敌是友、是大是小**。
因此，高水平人类棋手的开局实战遵循：**首翻我方子力据点化（Strongpoint Initiation）**：
- 无论翻出的首个己方棋子是何军阶（哪怕是军长、师长甚至司令），直接确立为该区域的战略据点；
- 棋手立刻以该据点为依托，占领四角行营或据点兵站，开始向相邻暗位辐射拓荒。

#### 2. 据点大子的“三向博弈主动权”（以据点军长翻 1/2/3 号相邻位为例）
依托据点大子翻开相邻暗子（如上方 1、右方 2、斜角 3），己方在所有可能结果中均占据绝对主动权：
- **情景 A：翻出敌方子力**：由于据点是大子（如军长），除司令/炸弹外全场压制。敌子刚被翻开当回合不可移动，据点大子下回合直接起步吃子，稳赚战果与先手（Tempo）；
- **情景 B：翻出敌方炸弹**：炸弹刚翻开处于瘫痪状态不可移动。**据点大子坚决不动**（绝不主动撞炸），主动权仍在己方手中。己方可继续翻开炸弹周围的其他相邻暗位，赌概率翻出己方小子就地碰掉拆弹；
- **情景 C：翻出己方子力（连锁扩圈与行营接力）**：翻出己方子力后，该新子力下回合直接移动进驻相邻的“B 行营”。新子力进营后免受攻击，并将侦察与翻棋辐射网络再次成倍外扩，形成“双星据点”或“行营铁三角”！

#### 3. 1000 局复盘前 20 手大数据实证
- **82.7% 进营率**：前 20 手出现的 8,503 次移动中，7,035 次（82.7%）是棋子进入行营（Into Camp）；
- **96.2% 邻营翻棋**：前 20 手出现的 10,615 次翻棋中，10,209 次（96.2%）紧贴行营相邻格发生；
- **进营子力自然随机**：排长 14.1%、工兵 13.8%、连长 13.7%、旅长 10.4%、团长 10.4%、营长 10.3%、师长 9.2%、炸弹 8.8%、军长 4.9%、司令 4.4%，与 25 子的自然概率完全拟合，彻底印证“翻到何子何子即据点”。
#### 4. 行营单向打击权与被翻子力战术死锁（Asymmetric Camp Strike Privilege）
棋手常言“翻棋多者胜率高”，其核心博弈机理在于**“翻出己方子力更多、进驻行营更多（明面控制据点占优）”**带来的压倒性主动权：
- **行营单向攻防不对称**：行营内棋子享有规则赋予的绝对免战权（外部任何棋子不能攻击行营）；但行营内的棋子却享有单向出营扑杀相邻敌子的主动权；
- **被翻出敌子的战术死锁态**：当在行营周围翻出敌方棋子时，该敌子当回合不可移动；到了下回合哪怕轮到敌方行动，由于行营免战，该敌子**在规则上绝不可能攻击行营内的子**！若其军阶小于行营子力，留在原地即成活靶，试图移动逃跑也会被行营子力在开阔地追杀；
- **50.1% 开局吃子源自出营扑杀**：大数据实证表明，在 1000 局前 20 手的全部 768 次吃子战斗中，**高达 50.1%（385 次）是由行营内棋子直接出营扑杀发起的**！谁能翻出更多己方棋子进驻行营，谁就拥有了遍布前线的单向火力发射台，将对手新翻出的暗子直接锁死为盘中餐。

#### 5. 占营比例胜率矩阵与量化影响因子（1000 局大数据实证）

**A. 占营控制比（我方:对方）与胜率矩阵**:

| 占营比 | 前 20 手决胜胜率 (样本量) | 前 40 手决胜胜率 (样本量) | 前 60 手决胜胜率 (样本量) | 局势特征 |
|---|---|---|---|---|
| **8 : 2** | 49.61% (178 局) | **62.96%** (73 局) | **63.16%** (57 局) | 压倒性控制，中残局胜率突破 63% |
| **7 : 3** | **56.67%** (230 局) | **53.61%** (152 局) | **54.46%** (159 局) | 显著优势，胜率稳固维持在 54%~57% |
| **6 : 4** | 46.86% (263 局) | 49.26% (367 局) | **51.98%** (353 局) | 微弱优势，随手数推移胜率由 47% 升至 52% |
| **5 : 5** | **50.00%** (342 局) | **50.00%** (274 局) | **50.00%** (234 局) | 均势基准，严格收敛于 50.00% |
| **4 : 6** | 53.14% (263 局) | 50.74% (367 局) | 48.02% (353 局) | 均势下半区 |
| **3 : 7** | 43.33% (230 局) | 46.39% (152 局) | 45.54% (159 局) | 劣势区间 |
| **2 : 8** | 50.39% (178 局) | **37.04%** (73 局) | **36.84%** (57 局) | 严重劣势，中残局胜率跌至 37% 以下 |

**B. 典型绝对占营对局胜率**:
- `5 营 : 2 营`：前 20 手胜率 **64.29%**，前 40 手胜率 **62.50%**；
- `6 营 : 2 营`：前 40 手胜率 **69.57%**；
- `6 营 : 3 营`：前 30 手胜率 **60.47%**；
- `5 营 : 3 营`：前 50 手胜率 **61.76%**；
- `2 营 : 1 营`：前 20 手胜率 **62.50%**。

**C. 量化影响因子与价值映射**:
- **边际胜率增量**: 中盘（前 40 手）每多占 1 个行营边际胜率 +0.23%；残局（前 60 手）每多占 1 营边际胜率 +0.81%；达到 8:2 压制时胜率绝对增量达 **+26.3%**；
- **引擎评估函数设计**: `w.camp_occ = 10.0`，即 1 个行营等价于 1 个连长，2 个行营等价于工兵/团长级别战略权重；叠加 `w.attack_camp` 单向出营扑杀权。

---

### 2.9 实践要点总结

✅ **核心原则**:
1. Mobility 是分层的（1 步 < 2 步 < 3 步潜力）
2. 不是所有格子价值相等，关键是识别战略节点
3. 空间优势可以通过算法量化
4. 图论工具可以帮助理解深层结构
5. 有时应该主动放弃空间换取其他优势

❌ **常见错误**:
1. 只关注棋子位置，忽视可达区域
2. 平均看待所有格子（每条铁路都一样）
3. 不知道何时该收缩、何时该扩张
4. 把行营当作普通格子对待

🎯 **正确的思维框架**:
```
空间思考的三个层面:

1. 局部层面：我这个棋子能动到哪？
   → Mobility-1 计算
   
2. 中期层面：我的棋子群能控制哪片区域？
   → Mobility-2/3 + 图连通性
   
3. 全局层面：我在整个棋盘上的位置如何？
   → 战略节点控制 + 空间优势评分
```

---

## 第 3 章：不完全信息下的威胁识别与响应 ⭐⭐⭐⭐⭐

### 3.1 威胁的本质：从物理攻击到信息压力

在完全信息博弈（如国际象棋）中，威胁通常被定义为："若对手走某步棋，我将损失子力/位置"。但在不完全信息军棋中，**威胁具有双重性质**：

**第一重威胁（物理层面）**：实际可见的子力交换风险。例如敌方明棋进攻我的未保护棋子。

**第二重威胁（信息层面）**：对手的行动在迫使我对暗子身份做出新假设，或在揭示关键信息的同时暴露我方弱点。

传统评估函数只捕捉第一重威胁，但**真正决定高手与普通玩家差异的是对第二重威胁的敏感度**。

#### 威胁的双重代价

当一个威胁发生时，其代价不仅包括物质损失，还包括：

```
总代价 = 物质损失 × 时间折扣 + 信息泄露惩罚 + 战略被动系数
```

**物质损失**: 标准棋子价值差（已方损失 - 敌方损失）

**时间折扣**: 威胁发生时的局面阶段因子
- 开局（移动数 < 20）：0.6（可以接受小亏换取速度）
- 中盘（20 ≤ 移动数 < 40）：1.0（正常成本）
- 尾盘（移动数 ≥ 40）：1.5（每一步都至关重要）

**信息泄露惩罚**: 被迫暴露强子或关键暗子身份的价值
```python
def calculate_info_leak_penalty(piece_id: int, board_state: GameState):
    """计算被迫暴露棋子身份的信息代价"""
    piece = board_state.get_piece(piece_id)
    
    if piece.is_hidden_before_move:
        uncertainty = belief_system.calculate_entropy(piece.pos)
        high_value_bonus = piece.value / 10.0 if piece.is_high_value else 0
        
        return uncertainty * 2.0 + high_value_bonus
    
    return 0.0
```

**战略被动系数**: 因应对威胁而失去的先手优势
- forced_tempo_loss: 被迫应招损失的 tempo 数
- position_degradation: 被迫后退/让出关键格点的比例

---

### 3.2 七类威胁分类系统

为了量化上述多重代价，我们将威胁分为七个层级，每一层对应不同的响应优先级和策略选择。

#### **T1 级：立即吃子威胁**（Immediate Capture）

**定义**：敌方明棋可在 1-2 步内吃掉我方明棋，且我方无有效防守手段。

**检测算法**：
```python
def detect_immediate_captures(board: Board, my_pieces, enemy_pieces):
    """检测 T1 级威胁"""
    threats = []
    
    for my_piece in my_pieces:
        if not my_piece.is_revealed:
            continue
            
        my_pos = my_piece.position
        
        for enemy_piece in enemy_pieces:
            if not enemy_piece.is_revealed:
                continue
                
            enemy_pos = enemy_piece.position
            
            capture_in_2_moves = can_capture_in_k_steps(
                attacker=enemy_piece,
                target=my_piece,
                steps=2,
                board=board
            )
            
            if capture_in_2_moves:
                value_diff = my_piece.value - enemy_piece.value
                
                # 考虑工兵挖雷的特殊性
                if is_mine(target=my_piece):
                    threat_strength = 5.0 + abs(value_diff) * 0.3
                elif is_explodable(target=my_piece):
                    threat_strength = 3.0 + abs(value_diff) * 0.5
                else:
                    threat_strength = max(0, -value_diff) * 1.2
                
                threats.append({
                    'type': 'T1',
                    'threatened_piece': my_piece.id,
                    'attacker': enemy_piece.id,
                    'estimated_time': count_min_steps(my_pos, enemy_pos),
                    'cost': value_diff,
                    'strength': threat_strength,
                    'confidence': 1.0
                })
    
    return sort_by_strenght(threats)
```

**响应策略**：
- **直接逃脱**：移动到安全格
- **反击威胁**：反向攻击敌方更脆弱的棋子
- **炸弹防御**：用可牺牲棋子进行拦截
- **弃子保大**：主动放弃低价值棋子保护核心战略目标

---

#### **T2 级：连环攻击威胁**（Chain Attack）

**定义**：单个敌方行动会依次触发多个威胁，形成组合拳效果。典型例子：敌炸弹炸掉护子后紧接着吃重要棋子，或利用行营循环调动制造"一步杀"。

**链式影响计算**：
```python
def calculate_chain_impact(threats: List[Threat], state: GameState) -> float:
    if not threats:
        return 0.0
    
    base_sum = sum(t['cost'] for t in threats)
    synergy_factor = 1.0 + (len(threats) - 1) * 0.3
    total_cost = base_sum * synergy_factor
    
    if has_flag_zone_threat(threats, state):
        total_cost *= 1.5
    
    return total_cost
```

**防御策略**：
- **阻断中间环节**：在敌人执行完整链条前插入干扰
- **同时保护多目标**：寻找能同时守护两个目标的"枢纽格点"
- **反链条预置**：提前部署 counter-threat
- **tempo 牺牲**：主动送一个可牺牲子来打乱节奏

---

#### **T3 级：旗区威胁**（Flag Zone Threat）

**定义**：敌方对我军旗保护区的任何入侵行为，无论当前子力得失如何，都具有最高战略优先级。

**旗区结构分析**：
```python
FLAG_ZONE_GEOMETRY = {
    'primary_approach_paths': [
        [(10, 0), (9, 0), (8, 0)],       # 左路直达
        [(10, 4), (9, 4), (8, 4)],       # 中路突破
        [(10, 8), (9, 8), (8, 8)]        # 右路渗透
    ],
    'safe_bunkers': [(9, 0), (9, 4), (9, 8)],
}

class FlagZoneDefender:
    def assess_vulnerability(self) -> VulnerabilityReport:
        vulnerable_paths = []
        
        for path in FLAG_ZONE_GEOMETRY['primary_approach_paths']:
            closest_enemy = find_closest_enemy_on_path(path)
            
            if closest_enemy and shortest_path_distance(closest_enemy.pos, self.my_flags) <= 3:
                vulnerable_paths.append({
                    'path': path,
                    'distance_to_flag': shortest_path_distance(closest_enemy.pos, self.my_flags),
                    'blocking_strength': calculate_defense_strength(path)
                })
        
        if not vulnerable_paths:
            return VulnerabilityReport(level='SAFE', score=0.0)
        
        max_threat_level = max(v['distance_to_flag'] for v in vulnerable_paths)
        overall_score = max_threat_level * 2.0 + (1 - avg_blockade) * 10
        
        return VulnerabilityReport(
            level='CRITICAL' if overall_score > 8 else 'WARNING' if overall_score > 5 else 'MODERATE',
            score=overall_score,
            recommended_actions=self._generate_defense_actions(vulnerable_paths)
        )
```

**防御决策树**：
```
旗区威胁等级评估：
├── SAFE (score < 5)
│   └── 维持现有兵力配置，适度向外扩展势力
├── MODERATE (5 ≤ score < 8)
│   ├── 将最弱路径上的格子加入重点监控
│   └── 预备一支快速反应部队
├── WARNING (8 ≤ score < 10)
│   ├── 立即调遣主力回防
│   └── 考虑用炸弹建立屏障
└── CRITICAL (score ≥ 10)
    ├── 不惜一切代价延缓敌前进
    └── 启动"焦土策略"
```

---

#### **T4 级：铁路入侵威胁**（Rail Invasion）

**定义**：敌方通过铁路网快速机动至我后方，绕过前沿防线直接威胁软目标（地雷、司令等）。

**铁路网络建模**：
```python
class RailNetworkAnalyzer:
    RAIL_GRID = {
        (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),     # 上行铁路
        (9, 2), (9, 3), (9, 4), (9, 5), (9, 6),     # 下行铁路
        (2, 0), (3, 0), (4, 0),                      # 左列铁路
        (2, 8), (3, 8), (4, 8),                      # 右列铁路
    }
    
    CENTRAL_JUNCTIONS = [(5, 0), (5, 4), (6, 0), (6, 4)]
    
    def analyze_invasion_risk(self, board_state: GameState) -> InvasionRiskReport:
        enemy_rail_controllers = []
        my_backline_vulnerable = []
        
        # 找出控制铁路的敌方棋子
        for pos in self.RAIL_GRID:
            piece = board_state.get_piece_at(pos)
            if piece and piece.is_enemy and piece.has_rail_movement():
                enemy_rail_controllers.append({
                    'position': pos,
                    'reach_radius': self.calculate_slide_range(pos),
                    'potential_targets': self.find_nearby_soft_targets(pos)
                })
        
        # 评估每条后端线路的脆弱性
        for pos in self.CENTRAL_JUNCTIONS:
            distance_to_my_flag = abs(pos[0] - self.my_flag_row)
            defense_strength = self.calculate_local_defense(pos)
            
            if distance_to_my_flag <= 3 and defense_strength < 3:
                my_backline_vulnerable.append({
                    'junction': pos,
                    'risk_level': min(10, distance_to_my_flag * defense_strength),
                    'recommended_response': self.generate_defense_for_junction(pos)
                })
        
        return InvasionRiskReport(
            control_points=enemy_rail_controllers,
            vulnerable_areas=my_backline_vulnerable,
            scenarios=self.predict_multi_pronged_attack(enemy_rail_controllers)
        )
```

**防御策略**：
1. **阻塞策略**：用中等价值棋子占据关键 junction
2. **诱捕策略**：故意留虚设防线引诱深入
3. **快速反应策略**：预先部署高 Mobility 棋子在交通枢纽附近

---

#### **T5 级：包围威胁**（Encirclement）

**定义**：敌方用较少兵力逐步压缩我活动空间，虽不立即吃子但使我最终陷入无路可走的绝境。

**检测算法**：
```python
def detect_encirclement_tendencies(board: Board, my_color: Color, k_history: int = 10):
    encirclements = []
    
    for piece in board.get_pieces(my_color):
        if not piece.is_revealed:
            continue
            
        current_mobility = len(mobility_2(board, piece.pos))
        mobility_history = get_mobility_history(piece.pos, window=k_history)
        
        if len(mobility_history) >= 3:
            slope = calculate_linear_regression_slope(mobility_history)
            
            if slope < -0.5:
                encirclements.append({
                    'piece_id': piece.id,
                    'encirclement_severity': abs(slope) * current_mobility,
                    'trend_direction': 'ACCELERATING' if slope < -1.0 else 'STEADY',
                    'escape_routes_remaining': count_viable_escape_paths(piece.pos)
                })
    
    return sorted(encirclements, key=lambda x: x['encirclement_severity'], reverse=True)
```

**突围策略库**：
| 突围类型 | 适用情况 | 风险等级 | 成功率 |
|---------|---------|---------|--------|
| **闪电突破** | 对方防线有一处明显薄弱点 | 高 | 60% |
| **声东击西** | 佯攻一侧，实际从另一侧突围 | 中 | 75% |
| **换子解围** | 用被包围子换取对方关键封锁子 | 中低 | 85% |
| **投降保留实力** | 确定必死，避免更大损失 | 低 | N/A |

---

#### **T6 级：诱导暴露威胁**（Information Trap）

**定义**：对手故意放置"诱饵棋子"，引诱我主动翻开暗子从而暴露身份，属于信息层面的心理战。

**诱饵检测**：
```python
def identify_potential_baits(board: Board, belief_system: BeliefTracker, candidate_positions: List[tuple]) -> List[BaitCandidate]:
    baits = []
    
    for pos in candidate_positions:
        piece = board.get_piece_at(pos)
        
        suspicious_patterns = [
            piece.value > 7 and not near_major_battlefield(pos),  # 高价值子异常活跃
            piece.is_exposed and count_enemies_within_distance(pos, radius=2) >= 2,  # 暴露在多次攻击下仍存活
            on_my_expected_capture_path(pos)  # 位于容易被"合理推测可以吃到"的路径上
        ]
        
        if any(suspicious_patterns):
            bait_probability = calculate_bait_likelihood(piece, board, opponent_history)
            
            if bait_probability > 0.6:
                baits.append({
                    'position': pos,
                    'apparent_value': piece.value,
                    'trap_complexity': estimate_trap_depth(),
                    'recommended_action': 'IGNORE or COUNTER_TRAP'
                })
    
    return baits

def calculate_bait_likelihood(piece: Piece, board: Board, opponent_history: Dict) -> float:
    from scipy.stats import binom
    
    prior = 0.15
    historical_tendency = opponent_history.get('bait_frequency', 0.1)
    abnormality_score = assess_position_abnormality(piece.pos, board)
    counter_kill_chance = estimate_counter_kill_rate(piece, board)
    
    likelihood_ratio = (historical_tendency * (1 - counter_kill_chance)) / \
                       ((1 - historical_tendency) * counter_kill_chance)
    
    posterior = prior * likelihood_ratio / (1 - prior + prior * likelihood_ratio)
    return min(1.0, posterior)
```

---

#### **T7 级：延迟威胁**（Deferred Threat）

**定义**：当前无害，但在未来某个时间点会被触发的隐患。例如敌方正在集结火力，或我的强子被逐步逼到角落失去战斗力。

**检测与预防**：
```python
def predict_deferred_threats(board: Board, horizon: int = 5) -> DeferredThreatList:
    deferred_threats = []
    
    for piece in my_pieces:
        future_trajectory = simulate_piece_trajectory(piece.pos, board, steps=horizon)
        
        if contains_fatal_zones(future_trajectory):
            deferred_threats.append({
                'affected_piece': piece.id,
                'threat_type': 'TRAPPED_PIECE',
                'time_to_crisis': find_first_entry_point(future_trajectory, FATAL_ZONES),
                'prevention_options': [
                    {'action': 'REPOSITION', 'pos': safe_exit_points(future_trajectory)},
                    {'action': 'PROVIDE_COVER', 'required_support_level': 3},
                    {'action': 'ABANDON_AS_BAIT'}
                ]
            })
    
    return deferred_threats
```

---

### 3.3 威胁响应优先级矩阵

面对多个并发威胁时，需要一个统一的优先级排序机制：

```
威胁响应优先级（按紧急程度降序排列）：
====================================================
Tier 1 - IMMEDIATE CRISIS (处理时限：< 2 步)
├── T3 级旗区威胁
├── T1 级吃子 + 我方高价值棋子
└── T2 级连环攻击导致≥2 个棋子面临损失

Tier 2 - URGENT PROTECTION (处理时限：3-5 步)
├── T1 级吃子 + 中等价值棋子
├── T4 级铁路入侵 + 靠近我方阵地
└── T5 级包围 + Mobility 降至≤2

Tier 3 - STRATEGIC ADJUSTMENT (处理时限：5+ 步)
├── T6 级诱导暴露
├── T4 级铁路入侵（距离较远）
└── 信息劣势积累导致的长期风险
```

**不确定性处理**：由于暗子存在，威胁检测本质上是不确定的。我们需要输出"**置信度分数**"而非绝对判断，置信度影响的是"响应力度"而非"是否响应"。宁可误报一千，不可漏掉一个真实的高阶威胁。

---

## 第 4 章：Tempo 经济学 ⭐⭐⭐⭐

### 4.1 Tempo 的本质：时间的货币化

在军棋翻棋中，**tempo（先手权）是最稀缺的资源**。传统棋盘游戏（如国际象棋）用"几步"衡量先后，但军棋有独特的 tempo 机制：

- 每回合只能执行一个动作
- 翻棋需要消耗 1 个 tempo
- 每个大子出动都需要 tempo
- 防守必须投入 tempo

**核心洞察**：优秀的玩家懂得在不同情况下权衡 tempo 的投入产出比。

---

### 4.2 Tempo 的四类流动

#### 1. Tempo Gain（获得先手）

当我方行动后，敌方被迫回应，我获得额外的 tempo：

```python
def calculate_tempo_gain(my_move: Move, resulting_state: GameState) -> float:
    """
    计算这一步带来的 tempo 收益
    
    例子：
    - 我用师长吃掉敌方旅长 → 获得 1 个 tempo
    - 我用工兵飞到安全位置并侦察 → 获得 1.5 个 tempo（信息 + 安全）
    """
    material_advantage = evaluate_material_change(my_move)
    positional_improvement = evaluate_positional_change(my_move)
    threat_created = evaluate_new_threats(resulting_state)
    
    return material_advantage * 0.3 + positional_improvement * 0.4 + threat_created * 0.3
```

#### 2. Tempo Loss（损失先手）

当我不利地行动后，失去 tempo：

```python
def calculate_tempo_loss(my_move: Move, enemy_response: Move) -> float:
    # 逃跑造成的损失
    if my_move.is_retreat:
        retreat_distance = distance_before_after(my_move)
        return retreat_distance * 0.5
    
    # 被动防守的损失
    if my_move.is_defensive:
        forced = my_move.is_forced  # 是否被迫
        return 1.0 if forced else 0.5
    
    # 白送一子的损失
    if my_move.sacrifices_piece():
        return 2.0 + evaluated_piece_value(my_move.lost_piece)
    
    return 0.0
```

#### 3. Forced Tempo（被迫花费）

当敌方威胁迫使我必须回应时：

```python
def calculate_forced_tempo(threats: List[Threat]) -> float:
    """
    计算为应对威胁必须付出的 tempo
    
    例如：
    - 敌方司令要吃我的军旗 → 我必须花 1 tempo 挡一下
    - 敌方工兵要挖雷 → 我必须花 1 tempo 守雷
    """
    total_forced = 0.0
    
    for threat in threats:
        if threat.priority == 'IMMEDIATE':
            total_forced += 1.0
        elif threat.priority == 'URGENT':
            total_forced += 0.5
    
    # 如果多个威胁同时存在，可能需要额外 tempo 协调
    if len(threats) > 2:
        total_forced += (len(threats) - 2) * 0.3
    
    return total_forced
```

#### 4. Tempo Reserve（储备先手）

有些行动虽然不直接获得 tempo，但为后续创造了条件：

```python
def calculate_tempo_reserve(positioning_move: Move) -> float:
    """
    计算这一步的潜在 tempo 价值
    
    例子：
    - 占领行营 → 下一步可以快速机动 (+0.5)
    - 连接两个棋子 → 协同作战效率提升 (+0.3)
    - 占据铁路枢纽 → 后续移动的灵活性增加 (+0.8)
    """
    future_mobility_boost = evaluate_mobility_improvement(positioning_move)
    coordination_bonus = evaluate_synergy(positioning_move)
    flexibility_increase = evaluate_path_optimization(positioning_move)
    
    return future_mobility_boost * 0.5 + coordination_bonus * 0.3 + flexibility_increase * 0.2
```

---

### 4.3 Tempo 的经济周期理论（基于 1000 局实战数据校准）

军棋的节奏类似于经济周期：**扩张期 → 高峰期 → 收缩期 → 低谷/决胜期**。
根据 1000 局官方实战数据的统计（详见 `reports/replays_1000_mining_report.md`），决胜局与和棋局展现出截然不同的生命周期与 Tempo 消耗特征：

- **决胜局生命周期**：平均仅 **89.5 手**（中位数 85 手）。超过 80% 的分胜负对局在 60~110 手之间决出，存在明显的 **80~90 手决胜窗口期**。
- **和棋局生命周期**：平均高达 **155.5 手**（中位数 150 手），最长对局达 272 手。当对局越过 100 手大关且未出现毁灭性破局时，子力流失与阵地僵化将使对局迅速滑向死锁与协议和棋。

#### 1. 扩张期（Turn 1-25，对应总步数 1-50 手）
- **特征**：双方抢占中路咽喉（实战首翻 `(7, 2)` 占 64.8%，`(6, 2)` 占 13.1%）与铁路枢纽。
- **Tempo 策略**：慷慨投资 tempo。翻棋提供位置与信息红利，但需防备孤军深入被围猎。
- **关键指标**：Mobility-3 增长率 > 0.2/回合，中心控制权占优。

#### 2. 高峰兑子期（Turn 26-45，对应总步数 51-90 手）⭐ **黄金决胜窗口**
- **特征**：双方大子全面接火，排兵布阵正面碰撞。炸弹、工兵与司令/军长展开斩首与反斩首博弈。
- **Tempo 策略**：精算每步 Tempo。实战中此阶段一旦一方形成 2 个以上绝对 Tempo 优势（如大子破阵、连续追击），将在 10~20 步内直接引发对手**防线崩溃与认输**。
- **关键指标**：Tempo Gain > 1.5，威胁传导链条连续不中断。

#### 3. 收缩/残局期（Turn 46-60，对应总步数 91-120 手）
- **特征**：双方剩余棋子降至 8~14 颗，若此时子力差 ≥ 1 师/大子，劣势方已进入投降折叠区；若双方均势且关键工兵阵亡，局势开始向死守演变。
- **Tempo 策略**：占优方加快拔旗/控场节奏；落后方若已无破防手段，Tempo 价值迅速趋近于 0。

#### 4. 僵持/低谷期（Turn 60+，对应总步数 120+ 手）
- **特征**：极高概率进入“理论和局”（死雷封路、双无工兵、行营避险），双方缺乏突破手段，进入 Zugzwang 平衡与步数倒数。
- **Tempo 策略**：避免多余无意义行棋，利用行营与安全铁路线维持 50 步无吃子或发起协议和棋。

---

### 4.4 Tempo 失衡的处理

当发现 tempo 不平衡时，有不同的应对方式：

#### 领先方的策略：保持压力
```python
def lead_maintenance_strategy(game_state: GameState) -> StrategyDecision:
    if game_state.tempo_advantage > 1.5:
        return {
            'mode': 'EXPANSIVE_PRESSURE',
            'actions': [
                'create_multiple_threats',  # 制造多个威胁迫使对方疲于奔命
                'expand_control_radius',    # 扩大控制范围
                'sacrifice_small_tempo_for_big_gains'  # 小额 tempo 投入换取大收益
            ]
        }
    return default_strategy()
```

#### 落后方的策略：扭转节奏
```python
def catch_up_strategy(game_state: GameState) -> StrategyDecision:
    if game_state.tempo_advantage < -1.0:
        return {
            'mode': 'RISK_TAKING',
            'actions': [
                'force_complex_combinations',      # 制造复杂局面
                'accept_small_losses_for_tempo',   # 接受小亏损换取 tempo
                'target_opponent_information_weakness'  # 攻击对方信息盲区
            ]
        }
    return conservative_defense()
```

---

### 4.5 Tempo 与信息的权衡

有时需要用 tempo 换取信息，这需要在动态评估：

```python
def should_spend_tempo_on_reconnaissance(pos: tuple, belief_system: BeliefTracker, current_tempo_balance: float) -> Decision:
    info_gain = calculate_information_gain(pos, belief_system)
    tempo_cost = 1.0
    opportunity_cost = estimate_best_alternative_use_of_tempo()
    
    net_value = info_gain - tempo_cost - opportunity_cost
    
    if current_tempo_balance > 0.5:  # 我们有 tempo 优势
        threshold = 0.3  # 较低门槛，愿意花钱买信息
    elif current_tempo_balance < -0.5:  # 我们 tempo 劣势
        threshold = 0.8  # 只有高价值时才侦察
    else:
        threshold = 0.5
    
    return net_value > threshold, f"Net Value: {net_value:.2f}, Threshold: {threshold:.2f}"
```

---

### 4.6 实战认输折叠线（Resignation Collapse & Terminal Horizon）⭐ 实战关键机制

1000 局对局挖掘揭示出一个极为重大的实战现象：**在决胜对局中，仅有 6.9% 是通过真正吃光或扛旗触发物理终局，高达 45.6% 均因玩家主动认输（Resign）或强退（15.0%）提前终局**。

#### 1. 认输折叠的本质
人类高水平棋手在遭遇毁灭性打击后，不会机械地把棋局走完，而是预见到未来 10~20 步内必败（Zugzwang 或无解将死），在心理和博弈论层面触发**决策折叠**：
- **子力断崖**：例如我方司令阵亡、军长被炸，而敌方尚存司令或双师，且我方无炸弹制衡。
- **空间失守**：军旗正面防守阵被突破，敌方工兵或大子已兵临城下，我方无有效防守步。
- **Tempo 绝望逆差**：每走一步都会白送一子（被迫动子暴露或被吃），进入绝对负收益循环。

#### 2. 对 AI 价值网络与搜索评估的启示
- **传统 AI 的盲区**：若价值函数（Value Head）仅仅被训练拟合“最终扛旗/吃光”的奖励，会导致模型在优势巨大时“不紧不慢”、在劣势绝望时产生“盲目走废步苟活”的幻觉。
- **折叠视界建模（Horizon Collapse）**：AI 评估函数在第 70~90 手黄金窗口期内，如果检测到胜率超过 95% 且敌方 Tempo 断崖，应直接将局面收敛为胜定状态；反之，若自评估陷入绝对不可逆折叠，应引导搜索策略进入极限拼刺刀博弈（寻求炸弹孤注一掷）。

```python
def check_resignation_collapse(game_state: GameState, player: int) -> bool:
    """
    基于实战大数据的认输折叠判定
    """
    my_power = evaluate_firepower_network(game_state, player)
    opp_power = evaluate_firepower_network(game_state, 1 - player)
    my_tempo = calculate_tempo_advantage(game_state, player)
    
    # 1. 绝对火力绝望：对手存大子而我方不仅无大子且无炸弹、无工兵
    has_threat_piece = my_power.bombs > 0 or my_power.max_rank >= opp_power.max_rank
    if not has_threat_piece and opp_power.max_rank >= RANK_DIVISION:
        if my_power.engineers == 0 and my_tempo < -2.0:
            return True  # 触发认输折叠线
            
    # 2. 旗区破防绝望：敌方大子/工兵已贴身逼近军旗且我方无法阻挡
    if is_flag_under_lethal_threat(game_state, player) and not can_defend_flag(game_state, player):
        return True
        
    return False
```

---

## 第 5 章：条件子力价值 ⭐⭐⭐⭐

### 5.1 为什么静态价值不够？

传统的棋子价值表是固定的：

```
司令 = 10.0
军长 = 9.5
师长 = 8.0
...
工兵 = 1.0
```

但这忽略了**情境的重要性**：

| 棋子 | 位置 | 静态价值 | 动态价值 | 原因 |
|------|------|---------|---------|------|
| 司令 | 中央行营 | 10.0 | 12.0 | 高度机动 + 可发挥最大威力 |
| 司令 | 角落被困 | 10.0 | 4.0 | 无法发挥作用 |
| 工兵 | 远离地雷 | 1.0 | 0.5 | 暂时无用 |
| 工兵 | 贴近地雷 | 1.0 | 3.0 | 即将挖雷，战略价值巨大 |
| 炸弹 | 保护军旗 | 7.0 | 9.0 | 不可替代的防守力量 |
| 炸弹 | 闲置后台 | 7.0 | 3.0 | 可能永远无法使用 |

**结论**：棋子价值必须根据**当前局面**动态调整。

---

### 5.2 动态价值的五个维度

对于每一个己方明子，我们需要计算：

```python
class ConditionalPieceValue:
    def __init__(self, piece, board, game_context):
        self.piece = piece
        self.board = board
        self.context = game_context
        
    def calculate_dynamic_value(self) -> float:
        base_value = PIECE_BASE_VALUES[self.piece.type]
        
        # 维度 1: 位置价值
        position_multiplier = self.evaluate_position_quality()
        
        # 维度 2: 威胁贡献度
        threat_contribution = self.evaluate_offensive_pressure()
        
        # 维度 3: 防守必要性
        defensive_necessity = self.evaluate_critical_role()
        
        # 维度 4: 机动潜力
        mobility_bonus = self.evaluate_future_mobility()
        
        # 维度 5: 特殊能力激活度
        ability_activation = self.evaluate_special_capabilities()
        
        dynamic_value = (
            base_value * position_multiplier * 
            (1 + threat_contribution) * 
            (1 + defensive_necessity) * 
            mobility_bonus * 
            ability_activation
        )
        
        return dynamic_value
```

#### 维度 1：位置质量评分

```python
def evaluate_position_quality(self) -> float:
    pos = self.piece.position
    
    # 中心性得分
    center_score = calculate_centralization(pos)
    
    # 安全性得分
    safety_score = 1.0 / (1 + count_enemies_nearby(pos))
    
    # 战略性得分
    strategic_score = sum(
        node_value for node_type, nodes in STRATEGIC_NODES.items()
        if pos in nodes['positions']
    )
    
    weighted_avg = center_score * 0.3 + safety_score * 0.4 + strategic_score * 0.3
    
    return normalize_to_0_2(weighted_avg)
```

#### 维度 2：威胁贡献度

```python
def evaluate_offensive_pressure(self) -> float:
    potential_threats = []
    
    # 我能威胁到什么？
    for target_pos in mobility_2(self.board, self.piece.position):
        target = self.board.get_piece_at(target_pos)
        if target and target.is_enemy:
            threat_strength = min(1.0, self.piece.attack_power / (target.attack_power + 0.1))
            potential_threats.append(threat_strength)
    
    if not potential_threats:
        return 0.0
    
    # 平均威胁强度 + 最大威胁加成
    avg_threat = sum(potential_threats) / len(potential_threats)
    max_threat = max(potential_threats)
    
    return avg_threat * 0.6 + max_threat * 0.4
```

#### 维度 3：防守必要性

```python
def evaluate_critical_role(self) -> float:
    # 这个棋子是否承担关键防守任务？
    
    flag_proximity = distance_to_flag(self.piece.position, direction='my')
    mine_proximity = distance_to_mines(self.piece.position, direction='enemy')
    
    # 靠近军旗且有攻击能力的棋子，防守价值极高
    if self.piece.can_attack() and flag_proximity <= 3:
        return 0.5 + (3 - flag_proximity) * 0.1
    
    # 靠近地雷线的工兵，防守价值高
    if self.piece.type == '工兵' and mine_proximity <= 2:
        return 0.8
    
    return 0.0
```

#### 维度 4：机动潜力

```python
def evaluate_future_mobility(self) -> float:
    current_mobility = len(mobility_1(self.board, self.piece.position))
    projected_mobility = len(mobility_3(self.board, self.piece.position))
    
    # 如果未来 mobility 增长空间大，说明当前位置好
    growth_ratio = projected_mobility / max(current_mobility, 1)
    
    # 有铁路通行的棋子 bonus
    rail_bonus = 1.5 if has_rail_access(self.piece.position) else 1.0
    
    mobility_factor = min(2.0, growth_ratio * 0.5 + 0.5)
    
    return mobility_factor * rail_bonus
```

#### 维度 5：特殊能力激活度

```python
def evaluate_special_capabilities(self) -> float:
    if self.piece.type == '工兵':
        # 工兵的价值取决于是否能飞/挖雷及当前局面期权
        can_fly = has_friendly_air_corridor(self.piece.position)
        can_dig_mine = is_adjacent_to_enemy_mine(self.piece.position)
        mine_option = calculate_engineer_mine_option(self.board, self.piece.color)
        
        base_mult = (1.5 if can_fly else 1.0) * (2.0 if can_dig_mine else 1.0)
        return base_mult * mine_option
    
    elif self.piece.type == '炸弹':
        # 炸弹的价值取决于是否有高价值目标与战略威慑力
        nearby_high_value = count_nearby_enemy_high_value(self.piece.position)
        return min(2.5, 1.0 + nearby_high_value * 0.6)
    
    elif self.piece.type in ['司令', '军长']:
        # 大子的价值取决于能否自由击杀与控场统治力
        free_killing_opportunities = count_free_kills(self.piece.position)
        return min(1.8, 1.0 + free_killing_opportunities * 0.15)
    
    return 1.0  # 普通棋子
```

---

### 5.3 实战子力价值校准专题（基于 1000 局官方对局实证）⭐

基于 1000 局官方终局与推演数据（详见 `reports/replays_1000_mining_report.md`），我们对传统军棋理论中多项主观假说进行了实证校准：

#### 1. 破除“单司令决定论”与二线火力网补偿系数
- **实战数据颠覆**：在 261 局明确分出司令阵亡先后的对抗中：
  - **先死司令一方告负**：130 局（**49.8%**）
  - **先死司令一方逆转获胜**：131 局（**50.2%**）
  - **结论**：胜负几乎完全对半开！传统评估模型中一旦司令死亡就将评估打入 -5.0 甚至 -10.0 的做法是严重失真的。
- **二线火力网接管模型（Secondary Firepower Resilience）**：
  先死司令能否逆转，完全取决于**二线梯队（军长、双师长、双炸弹）的完整性**。当司令兑掉对方大子或探雷阵亡后，若己方军长仍在且双炸弹在手，对手的司令亦受到严密牵制，我方火力完全不落下风。
  
```python
def calculate_commander_loss_resilience(board: Board, color: Color) -> float:
    """
    计算司令阵亡后的局面韧性系数（0.0 ~ 1.0）
    1.0 表示二线梯队完好，战力完全可平替；0.0 表示梯队断层，直接崩盘
    """
    my_corps = count_piece(board, color, '军长')
    my_divisions = count_piece(board, color, '师长')
    my_bombs = count_piece(board, color, '炸弹')
    
    # 二线火力打分（军长权重 0.45，每颗师长 0.20，每颗炸弹 0.20）
    firepower_score = my_corps * 0.45 + my_divisions * 0.20 + my_bombs * 0.20
    resilience = min(1.0, firepower_score)
    
    # 动态惩罚衰减：若 resilience 接近 1.0，司令阵亡惩罚衰减 60%
    return resilience
```

#### 2. 工兵“挖雷期权”非线性折现模型（Engineer Mine Option Pricing）
- **实战发现**：
  - 开中盘双方各有 3 颗工兵，此时单个工兵战死仅损失约 1.5~2.0 分；
  - 但当进入残局双方仅剩 1 兵时，若敌方军旗被 2 颗地雷死死封锁，该工兵是**破防获胜的唯一充分必要条件**。工兵阵亡将直接使胜局（1.0）折叠为理论和局（0.0）；
  - 反之，当敌方地雷已被炸光/挖光，工兵的“挖雷期权”归零，其战力仅相当于比排长更弱的弱子（0.8分），可果断用于兑换、堵路或刺杀敌方炸弹。

```python
def calculate_engineer_mine_option(board: Board, color: Color) -> float:
    """
    计算工兵的挖雷期权乘数
    """
    opp_color = 1 - color
    alive_opp_mines = count_piece(board, opp_color, '地雷')
    alive_my_engineers = count_piece(board, color, '工兵')
    
    if alive_opp_mines == 0:
        return 0.8  # 雷已挖完，期权归零，工兵仅值普通肉子
        
    if alive_my_engineers == 1:
        # 独苗工兵，且敌方尚有地雷护旗，期权溢价极高
        return 2.5 + alive_opp_mines * 0.5  # 溢价可达 3.0~4.0 倍
        
    return 1.2 + (alive_opp_mines / max(1, alive_my_engineers)) * 0.4
```

#### 3. 炸弹实战消耗规律与“小子贴身拆弹”定式
- **实战数据**：炸弹与司令/军长同归于尽仅占 **14.2%**，高达 **47.5%** 的炸弹被消耗在连长、排长、团长等中小子力上。
- **棋理机制矫正**：
  - 翻棋规则硬约束：**不可攻击暗子**。因此中小子绝非所谓“踩炸弹”或“炸弹主动清暗子”。
  - 真实博弈逻辑：**“行营小子摸奖，就地贴身拆弹”**。防守方利用排长、连长等廉价子力进驻绝对安全的行营；一旦相邻暗子翻出敌方炸弹，由于暗子翻开当回合不可移动，行营内就绪的廉价小子在随后的轮次中**主动出营扑撞拆弹**！以 18 分的排长换掉敌方 52 分唯一的战略核武，获得巨大的物质与战略净收益。
- **策略与估值启示**：
  - **暗炸威慑溢价**：炸弹在未翻开（暗子）或未暴露时战略威慑力极高（隐含估值 8.0~9.0）；
  - **暴露即折价**：一旦炸弹在敌方控阵或行营周围被翻开，由于无法当回合移动，极易被敌方廉价小子就地拆解，其动态价值直接雪崩断崖；
  - AI 翻棋决策应严格评估行营周边的落点风险，避免自身战略炸弹在无护卫情况下孤立暴露于敌方行动圈内。

---

### 5.4 整体局面价值评估

除了单个棋子的动态价值，还需要计算**双方在当前阶段的整体价值对比**：

```python
def evaluate_phase_based_value(game_state: GameState) -> PhaseValueAssessment:
    turn = game_state.turn_count
    
    if turn < 20:  # 开局阶段
        values = evaluate_opening_phase_values(game_state)
    elif turn < 40:  # 中盘阶段
        values = evaluate_midgame_values(game_state)
    else:  # 残局阶段
        values = evaluate_endgame_values(game_state)
    
    return values
```

**开局阶段**：更看重 Mobility 和控制中心的能力  
**中盘阶段**：更看重攻击机会和防守稳定性  
**残局阶段**：更看重定式完成度和 Zugzwang 优势

---

### 5.4 实践要点总结

✅ **核心原则**:
1. 棋子价值是动态的，不是静态的
2. 位置决定价值上限
3. 同一棋子在不同阶段价值不同
4. 特殊能力（工兵飞行、炸弹同归于尽）需要针对性评估
5. 信息不对称下，动态价值更不准确但仍需估算

❌ **常见错误**:
1. 盲目相信固定价值表
2. 忽视棋子位置对价值的巨大影响
3. 不考虑阶段差异（用开局思路评估残局）
4. 低估暗子的不确定性

🎯 **正确的思维框架**:
```
计算棋子价值三步法:

Step 1: 基础价值是多少？
  → 查固定价值表

Step 2: 当前局面对这个棋子有利吗？
  → 位置好吗？能发挥作用吗？
  → 乘上位置和质量系数

Step 3: 这个棋子对未来有什么意义？
  → 将来会变强还是变弱？
  → 加上/减去机动潜力和特殊能力
```

---

**第 3-5 章完成标志**：建立了完整的战术基础三层架构——威胁识别、Tempo 管理、条件价值评估。这三章是连接信息/空间基础与高层战略计划的核心桥梁。

---

## 第 6 章：战略计划层 ⭐⭐⭐⭐⭐

### 6.1 从战术计算到战略规划

前三章建立了**信息推理、空间控制、威胁识别、Tempo 管理、条件价值**五大基础概念。但这些仍然属于"战术层面"——关注的是单个行动的评估。

**战略层面的核心问题**是：

> "在当前局面下，我应该追求什么**总体目标**？这个目标如何分解为可执行的子步骤？"

#### 战略计划的四个特征

1. **时间跨度长**：涉及多个回合（通常 > 5 步）
2. **目标导向性**：有明确的中间里程碑
3. **灵活性**：可以根据对手反应调整细节
4. **资源协调**：需要调动多个棋子协同作战

---

### 6.2 战略计划库的设计

我们需要建立一个**计划库**，包含常见的战略模式：

```python
class StrategicPlanLibrary:
    """战略计划库"""
    
    PLANS = {
        'quick_flag_attack': QuickFlagAttackPlan(),      # 快速攻旗
        'positional_strangle': PositionalStranglePlan(),  # 位置绞杀
        'material_sacrifice': MaterialSacrificePlan(),    # 弃子突破
        'defensive_counter': DefensiveCounterPlan(),      # 防守反击
        'railway_invasion': RailwayInvasionPlan(),        # 铁路入侵
        'mine_clearance': MineClearancePlan(),            # 排雷进攻
        'tempo_accumulation': TempoAccumulationPlan(),    # tempo 积累
        'information_domination': InformationDominationPlan()  # 信息压制
    }
    
    def select_best_plan(self, game_state: GameState) -> StrategicPlan:
        """根据当前局面选择最合适的计划"""
        plan_scores = []
        
        for plan_name, plan_template in self.PLANS.items():
            score = plan_template.evaluate_fit(game_state)
            confidence = plan_template.get_confidence(game_state)
            
            plan_scores.append({
                'name': plan_name,
                'score': score,
                'confidence': confidence,
                'plan': plan_template
            })
        
        # 按综合评分排序
        plan_scores.sort(key=lambda x: x['score'] * x['confidence'], reverse=True)
        
        return plan_scores[0]['plan'], plan_scores
    
    def adjust_plan_mid_execution(self, original_plan: StrategicPlan, 
                                   unexpected_event: Event) -> AdjustedPlan:
        """执行过程中遇到意外事件时的计划调整"""
        if original_plan.can_absorb_event(unexpected_event):
            return original_plan.continue_with_modification(unexpected_event)
        else:
            # 切换到 Plan B
            return self.find_alternative_plan(original_plan.objective, unexpected_event)
```

#### 典型战略计划示例

##### 计划 A：快速攻旗（Quick Flag Attack）

```python
class QuickFlagAttackPlan(StrategicPlan):
    """
    战略目标：在最短时间内突破敌方雷阵并攻击军旗
    
    适用条件:
    - 我方已经掌握足够的信息（敌人工兵位置基本明确）
    - 我方有至少 2 个工兵处于活跃状态
    - 敌方前方防线空虚或有明显漏洞
    
    阶段分解:
    Phase 1 (Turn 1-15): 侦察与定位
      - 侦察工兵潜在位置
      - 确定雷阵结构
      - 找到最佳突破口
    
    Phase 2 (Turn 16-25): 开辟通道
      - 用工兵清理一条安全路径
      - 保护工兵不被敌方大子拦截
      - 建立前沿基地
    
    Phase 3 (Turn 26-40): 最终突击
      - 集中所有剩余工兵于突破口
      - 一次性投入 ≥ 3 个工兵同时挖雷
      - 确保至少有 1 个工兵能接触军旗
    
    Phase 4 (Turn 40+): 决胜
      - 用最后一个工兵挖掉军旗前的最后一道防线
      - 完成击杀
    ```

**执行监控指标**:
- 工兵活跃度：≥ 2 个工兵在 Turn 15 前进入前线区域
- 路径清晰度：至少有 1 条通往军旗的路径熵 < 0.3
- 速度要求：Turn 30 前必须完成 Phase 2

##### 计划 B：位置绞杀（Positional Strangle）

```python
class PositionalStranglePlan(StrategicPlan):
    """
    战略目标：通过逐步压缩敌方空间，使其陷入动弹不得的境地
    
    适用条件:
    - 我方拥有空间优势（Mobility-3 比值 > 1.5）
    - 敌方暗子较多且位置不明
    - 我方可控节奏，不需要急于求战
    
    阶段分解:
    Phase 1: 控制中心枢纽
      - 抢占中央行营 (5,2)/(8,2)
      - 封锁铁路交叉点
    
    Phase 2: 制造多重威胁
      - 同时在左、中、右三路施加压力
      - 迫使敌方分散兵力
    
    Phase 3: 逐个击破
      - 利用敌方薄弱点建立桥头堡
      - 逐步扩大控制区域
    
    Phase 4: 致命包围
      - 完全切断敌方主力机动路线
      - 等待敌方自乱阵脚
    ```

---

### 6.3 计划分解与执行策略

一个宏观的战略计划必须分解为具体的可执行动作：

```python
def decompose_plan_into_actions(plan: StrategicPlan, 
                                 game_state: GameState) -> List[ExecutionStep]:
    """
    将战略计划分解为战术动作序列
    
    例如：
    战略目标："快速攻旗"
    ↓ 分解为
    [
      Step 1: 侦察 (5,2) 位置的暗子
      Step 2: 移动工兵 B 至 (7,3)
      Step 3: 用工兵 C 炸开 (8,5) 疑似雷区
      ...
    ]
    """
    steps = []
    
    # 根据计划类型生成初始动作列表
    if isinstance(plan, QuickFlagAttackPlan):
        steps.extend(plan.generate_scouting_priorities(game_state))
        steps.extend(plan.identify_bombardier_candidates(game_state))
    
    # 按优先级排序
    steps.sort(key=lambda s: s.priority_score, reverse=True)
    
    # 添加依赖检查
    for i, step in enumerate(steps):
        dependencies = step.identify_dependencies(steps[:i])
        step.set_preconditions(dependencies)
    
    return steps
```

#### 执行中的动态调整

计划不是一成不变的，需要根据实时反馈进行调整：

```python
class PlanExecutor:
    def __init__(self, current_plan: StrategicPlan):
        self.plan = current_plan
        self.execution_history = []
        
    def execute_step(self, step: ExecutionStep, game_state: GameState) -> ExecutionResult:
        """执行单个战术步骤"""
        result = self._apply_action(step.action, game_state)
        
        # 记录执行情况
        self.execution_history.append({
            'step': step.id,
            'result': result,
            'actual_outcome': result.outcome,
            'expected_outcome': step.expected_outcome
        })
        
        # 检查结果是否符合预期
        if not self._is_on_track(result):
            return self._replan()
        
        return result
    
    def _replan(self) -> RepositioningDecision:
        """当执行偏离预期时重新规划"""
        deviation_analysis = self.analyze_deviation()
        
        if deviation_analysis.severity < 0.3:
            # 小偏差：微调后续步骤
            return self._fine_tune_remaining_steps()
        
        elif deviation_analysis.severity < 0.7:
            # 中等偏差：切换到备选方案
            return self._switch_to_backup_plan()
        
        else:
            # 严重偏差：完全重新制定计划
            return self._full_replan()
```

---

### 6.4 Plan B 机制

每个主要战略计划都应该有备选方案：

```python
class StrategicPlanWithFallbacks(StrategicPlan):
    def __init__(self):
        self.primary_plan = QuickFlagAttackPlan()
        self.backup_plans = {
            'info_reveal_failure': MaterialSacrificePlan(),      # 侦察失败 → 弃子试探
            'engineer_loss': DefensiveCounterPlan(),             # 工兵损失 → 转为防守
            'tempo_setback': TempoAccumulationPlan(),            # tempo 落后 → 积累优势
            'flag_defense_strength': PositionalStranglePlan()    # 攻旗困难 → 位置消耗
        }
    
    def handle_fallback_trigger(self, trigger_type: str, context: Dict) -> StrategicPlan:
        """处理 Plan B 触发条件"""
        if trigger_type in self.backup_plans:
            backup = self.backup_plans[trigger_type]
            
            # 验证 Backup 是否在当前局面可行
            if backup.is_feasible(context):
                return backup.with_context_adaptation(context)
        
        # 如果标准 Plan B 不可行，启动紧急预案
        return self.emergency_protocol(context)
    
    def emergency_protocol(self, crisis: CrisisEvent) -> EmergencyPlan:
        """危机情况下的应急方案"""
        if crisis.type == 'FLAG_IMMINENT_DEATH':
            return UltimateDefensePlan()  # 终极防守
        
        elif crisis.type == 'MAJOR_PIECE_LOST':
            return AttritionWarPlan()  # 消耗战
        
        else:
            return DesperationAttackPlan()  # 孤注一掷
```

---

### 6.5 基于贝叶斯更新的计划适应性

高级的计划系统应该能够根据新的信息实时更新概率：

```python
class BayesianPlanUpdater:
    def __init__(self, plans: List[StrategicPlan], prior_probabilities: Dict[str, float]):
        self.plans = plans
        self.prior = prior_probabilities
        self.posterior = prior_probabilities.copy()
        
    def update_with_observation(self, observation: Observation):
        """根据观察更新各计划的成功概率"""
        
        likelihoods = {}
        for plan_name, plan in zip(self.plans.keys(), self.plans):
            # P(observation | plan is correct)
            likelihood = plan.predict_likelihood(observation)
            likelihoods[plan_name] = likelihood
        
        # Bayes 公式
        evidence = sum(
            self.prior[name] * likelihoods[name]
            for name in self.prior
        )
        
        for plan_name in self.prior:
            self.posterior[plan_name] = (
                self.prior[plan_name] * likelihoods[plan_name] / evidence
            )
        
        # 归一化
        total = sum(self.posterior.values())
        self.posterior = {k: v/total for k, v in self.posterior.items()}
        
        return self.posterior
    
    def get_best_current_plan(self) -> Tuple[StrategicPlan, float]:
        """获取当前最优计划及其置信度"""
        best_plan = max(self.posterior.items(), key=lambda x: x[1])
        return self.plans[best_plan[0]], best_plan[1]
```

---

### 6.6 实践要点总结

✅ **核心原则**:
1. 战略计划需要有明确的时间跨度和里程碑
2. 每个主计划都应该有备份方案（Plan B）
3. 计划不是固定的，需要动态调整
4. 信息更新会改变计划的可行性，要用贝叶斯更新
5. 好的战略是"框架 + 弹性"的结合

❌ **常见错误**:
1. 没有长远规划，只看眼前得失
2. 过于僵化的执迷计划，忽视变化
3. 忽略 Plan B 的重要性
4. 计划分解不够细致，无法落地执行
5. 用单一指标（如子力价值）做决策

🎯 **正确的思维框架**:
```
战略思考的完整链条:

1. 局面诊断：当前处于什么阶段？优势在哪里？弱势是什么？
   → 识别机会点和风险点

2. 目标设定：我最想达成的战略目标是什么？
   → 快速攻旗？消磨对方？还是拖入残局？

3. 计划选择：有哪些可行的计划可以达成这个目标？
   → 从计划库中选择最匹配的模板

4. 任务分解：如何将这个目标分解为可执行的小步骤？
   → 创建详细的行动清单

5. 风险监控：哪些情况会导致我的计划失败？
   → 准备 Plan B、C、D

6. 执行与调整：实际执行后及时反馈修正
   → 观察→判断→调整的循环
```

---

## 第 7 章：残局定式库 ⭐⭐⭐⭐

### 7.1 残局的重要性

残局占整盘游戏的比例虽然不大，但**胜负往往在此决出**。很多情况下：

- 中盘看似均势，残局一方多一个工兵就必胜
- 开局犯的错误可能在中盘被掩盖，但在残局暴露
- 高手与新手的差距在残局的定式知识上体现得最明显

**残局的核心特征**：
1. 棋子数量少（通常 ≤ 10 个）
2. 路径更清晰，不确定性降低
3. Tempo 变得极其重要（1 步之差决定胜负）
4. 存在大量"必胜/必和"的定式

---

### 7.2 残局分类体系

我们将残局分为三大类：

#### Type I: 子力优势型残局（Material Advantage Endgame）

一方明显拥有更多或更强的棋子。

**典型情况**:
- 多子胜定（如 2 工兵 vs 1 地雷）
- 大子优势（如司令 vs 旅长）
- 炸弹数量差异

**求解方法**: 使用 Minimax 搜索 + Alpha-Beta 剪枝，深度达到终局

#### Type II: 位置优势型残局（Positional Advantage Endgame）

子力接近，但一方占据更有利的位置。

**典型情况**:
- 工兵贴近军旗 vs 工兵远离
- 占领行营 vs 被困角落
- 铁路控制权差异

**求解方法**: 结合位置评分 + 有限深度搜索

#### Type III: 理论和局型残局（Theoretical Draw Endgame）

双方实力相当，理论上可以逼和。

**典型情况**:
- 同等数量的对称残局
- 一方死守雷阵，另一方无法突破
- Zugzwang 平衡态

**求解方法**: 模式匹配 + 防御定式

---

### 7.3 必胜与必和定式库（Endgame Tablebase）

基于 1000 局官方实战中 301 局和棋与 456 局认输局面的深挖，我们建立涵盖胜局突破与死锁和局的高频定式库：

#### 定式 1: 双工兵 vs 单雷（兵多雷寡破阵定式）

```
局面:
- 红方：工兵 A(6,2), 工兵 B(6,4)
- 黑方：地雷 (8,4)，无其他机动大子支援
- 结果：红方必胜 (Red wins in 3-5 moves)

走法路径:
1. 红：工兵 A 前压掩护，工兵 B 沿铁路线机动切入 (7,4)
2. 黑：无子可应或无效行棋
3. 红：工兵 B(7,4) → (8,4) 挖雷成功，直接打开军旗通路
关键洞察：工兵数量 > 护卫力量且通路受控，为破防必胜定式。
```

#### 定式 2: 司令 vs 旅长 + 炸弹（博弈牵制态）

```python
# 复杂的非完全信息残局，需要动态牵制
ENDGAME_TABLEBASE = {
    'marshal_vs_brigadier_plus_bomb': {
        'draw_probability': 0.65,
        'win_probability': 0.25,
        'loss_probability': 0.10,
        'optimal_strategy': 'avoid_direct_combat_until_bomb_is_exposed'
    }
}
```

#### 定式 3: 工兵单骑破阵（无护卫军旗）

```
局面:
- 敌方大子尽失，军旗周围地雷已被排空
- 我方工兵在铁路上且无阻挡
- 必胜路径：工兵沿铁路直达底线切入大本营拔旗。
```

#### 定式 4: 双无工兵死和定式（Dual No-Engineer Deadlock）⭐ 理论和棋最高频形态

这是 1000 局实战中 **301 局和棋最经典的成因之一**：

```
局面特征:
1. 双方工兵均已全军覆没（己方工兵=0 且 敌方工兵=0）；
2. 守方军旗前方或侧翼存在未被排空的暗雷/明雷保护；
3. 任何常规棋子（包括司令、军长）撞雷均阵亡同归，无法拔除地雷。

理论结果: 严格理论必和 (Theoretical Dead Draw, Value = 0.0)

实战指导意义:
- 进攻方警示：哪怕手握司令、军长巨大子力优势，一旦失去工兵且敌旗有雷，绝对无法获胜。切忌急躁盲目撞雷白送大子，应主动转入协议和棋。
- 防守方翻盘契机：当处于劣势时，集中全部力量（包括用炸弹、小兵同归于尽）消灭敌方最后一名工兵，即可将败局强制锁定为平局！
```

#### 定式 5: 行营避险死守定式（Camp-Camp Stall Draw）⭐ 规则逼和定式

实战中 265 局协议和棋与 31 局限步判和的核心战术：

```
局面特征:
1. 劣势方仅存 1~2 颗弱子（如团长、排长），但成功进驻中心行营或后方安全行营；
2. 优势方虽握有全场最大子（如单军长或单师长），但受限于“行营内棋子免受攻击”规则，无法直接入营吃子；
3. 劣势方利用相邻两个行营或营外安全点来回腾挪（Camp-to-Camp Shuttle），避免被封死。

理论结果: 逼和 (Forced Draw by 50-Move Rule or Agreement)

实战要点:
- 优势方如果不能调动两颗以上棋子形成“守株待兔”的双向截断，单大子无法破防；
- 劣势方应精确计算步数，坚决不离开安全营区，直至触发 50 步无吃子判和。
```

#### 定式 6: 军旗贴身绝杀定式（Flag-Adjacent Cutoff Checkmate）⭐ 认输折叠主要来源

实战中 45.6% 认输对局的典型终局点：

```
局面特征:
1. 优势方大子（司令/军长）已进驻敌方大本营正上方兵站或咽喉行营；
2. 守方已无炸弹，亦无能抗衡的大子救援；
3. 优势方工兵或副将已就位，下一手即可强行扛旗或消灭最后防守子。

博弈论意义:
- 绝望死棋：守方任何移动都无法改变军旗被拔或被全歼的命运；
- 人类棋手在此局面通常不会走完最后 1-2 步，而是直接点击认输。AI 模型在搜索到此类状态时，应将其视为终端必胜态（Terminal Win, Value = 1.0）。
```

#### 定式 7: 单大子巡场扫荡定式（Single Heavy Piece Patrol）⭐ 铁路 Zugzwang

```
局面特征:
1. 优势方拥有全场唯一存活的超大子（司令或军长），且铁路网络畅通无阻；
2. 劣势方大子全灭，仅剩若干散落的小子（连排工兵）；
3. 优势大子在铁路上大范围巡弋，控制所有路口。

战术机制:
- 劣势方小子只要出站踏上铁路，必在 1 步内被全图机动的大子精准击杀；
- 若不出站，则在兵站内被步步蚕食逼入死角，形成全面 Zugzwang；
- 人类棋手在此阶段的逃跑/投降率超过 90%。
```

---

### 7.4 Zugzwang（逼走困境）识别

Zugzwang 是国际象棋中的一个重要概念，在军棋残局中同样关键：

**定义**：当前轮到某方走棋，但任何合法移动都会使该方的局面变差。

```python
def detect_zugzwang(game_state: GameState, color: Color) -> ZugzwangReport:
    """
    检测是否存在 Zugzwang
    
    判断方法:
    1. 枚举所有可能的移动
    2. 评估每个移动后的局面值
    3. 如果所有移动都导致局面值下降 → 存在 Zugzwang
    """
    all_moves = generate_all_legal_moves(color, game_state)
    
    if not all_moves:
        return ZugzwangReport(
            exists=False,
            reason='no_legal_moves',  # 无路可走是另一种失败
            severity=1.0
        )
    
    current_value = evaluate_position(game_state, color)
    
    worst_next_value = float('inf')
    
    for move in all_moves:
        simulated_state = apply_move(game_state, move)
        next_value = evaluate_position(simulated_state, color)
        
        worst_next_value = min(worst_next_value, next_value)
    
    if worst_next_value < current_value:
        return ZugzwangReport(
            exists=True,
            severity=current_value - worst_next_value,
            best_avoided_move=find_move_that_minimizes_damage(all_moves)
        )
    
    return ZugzwangReport(exists=False, reason='no_degradation')
```

**实战案例**:

```
局面:
- 红方：军长 (5,2)，只有这一个可动棋子
- 黑方：司令在 (5,0) 虎视眈眈
- 红方其他棋子都被锁死

分析:
红方军长有三个选择：
1. (5,2) → (4,2): 让出中央控制点 → 局势恶化
2. (5,2) → (6,2): 后退 → 失去活动能力  
3. (5,2) → (5,3): 侧移 → 可能被围攻

无论怎么走都会变差 → Zugzwang!
```

**利用 Zugzwang 的策略**:
- 如果我是优势方：主动制造 Zugzwang
- 如果我是劣势方：尽力避免陷入 Zugzwang

---

### 7.5 残局 Tempo 计算

残局中每一步的价值都被放大，需要精确计算：

```python
def calculate_endgame_tempo(game_state: GameState) -> TempoAssessment:
    remaining_moves = estimate_remaining_turns(game_state)
    critical_path_length = shortest_path_to_objective(game_state)
    
    my_tempo_advantage = len(my_available_fast_moves()) - len(enemy_fast_moves())
    
    # Tempo 临界点
    if my_tempo_advantage >= critical_path_length:
        return TempoAssessment(
            status='WINNING_POSITION',
            recommended_action='accelerate',
            priority_level='MAXIMUM'
        )
    
    elif my_tempo_advantage <= -critical_path_length:
        return TempoAssessment(
            status='LOSING_POSITION', 
            recommended_action='delay_and_complicate',
            priority_level='CRITICAL'
        )
    
    else:
        return TempoAssessment(
            status='COMPETITIVE',
            recommended_action='maintain_balance',
            priority_level='HIGH'
        )
```

**例子**: 残局还剩 10 步，我需要 5 步到达军旗，对方需要 6 步。
- 我的 tempo 优势 = 1 步
- 临界长度 = 5 步
- 结论：**我有 tempo 优势**，应该加速推进

---

### 7.6 残局数据库设计

为了支持高效的残局求解，我们需要构建**残局表库**（类似国际象棋的 endgame tablebases）：

```python
class EndgameTablebase:
    def __init__(self):
        self.table = {}  # (piece_positions, turn) -> outcome
    
    def store_winning_position(self, state: GameState, depth: int):
        """存储必胜局面及获胜所需步数"""
        key = self._state_to_key(state)
        self.table[key] = {
            'outcome': 'WIN',
            'moves_to_win': depth,
            'best_line': self._extract_best_line(state)
        }
    
    def store_drawn_position(self, state: GameState):
        """存储理论和局局面"""
        key = self._state_to_key(state)
        self.table[key] = {
            'outcome': 'DRAW',
            'drawing_method': self._identify_draw_mechanism(state)
        }
    
    def query(self, state: GameState) -> TablebaseResponse:
        """查询某个局面的理论结果"""
        key = self._state_to_key(state)
        
        if key in self.table:
            entry = self.table[key]
            return TablebaseResponse(
                outcome=entry['outcome'],
                guidance=self._generate_guidance(entry, state)
            )
        
        return TablebaseResponse(
            outcome='UNKNOWN',
            guidance=self._fallback_heuristic(state)
        )
```

**已计算的经典残局模式**:
- N 工兵 vs M 地雷（N,M ≤ 5）
- 1 大子 + K 工兵 vs 军旗守卫阵
- 炸弹数量不对称残局
- 纯铁路残局（仅大子在铁路上机动）

---

### 7.7 实践要点总结

✅ **核心原则**:
1. 残局有大量已知的必胜/必和定式，应该学习和记忆
2. Tempo 在残局中被高度放大，需要精确计算
3. Zugzwang 是一个强大的武器，要善于发现和利用
4. 残局表库可以大幅减少计算量
5. 信息优势在残局中仍然是关键

❌ **常见错误**:
1. 中盘就过早兑子进入残局（除非确定有利）
2. 忽视工兵的特殊价值
3. 误判定式（把必和当成必胜）
4. Tempo 计算错误导致慢一步

🎯 **正确的思维框架**:
```
残局处理的完整流程:

1. 识别残局类型:
   - 子力优势？位置优势？理论和局？
   
2. 查找定式库:
   - 是否有现成的必胜/必和模式？
   
3. 如果没有定式：
   - 进行有限深度的 Minimax 搜索
   - 结合位置评分函数
   
4. 特殊考虑:
   - 是否存在 Zugzwang?
   - Tempo 对比如何？
   - 是否需要避和或求胜？
   
5. 执行:
   - 按照定式或搜索结果走棋
   - 监控对手的应对
```

---

## 第 1-7 章全文总结

至此，《军棋翻棋棋理体系 v2.0》已完成全部七章的撰写，形成了完整的认知框架：

| 章节 | 主题 | 核心价值 |
|------|------|---------|
| Ch1 | 不完全信息推理 | 理解暗子的本质和信息的价值 |
| Ch2 | 空间与势力控制 | 学会衡量位置和移动能力 |
| Ch3 | 威胁体系 | 识别不同类型和等级的威胁 |
| Ch4 | Tempo 经济学 | 懂得时间和节奏的权衡 |
| Ch5 | 条件子力价值 | 动态评估棋子在不同情境下的价值 |
| Ch6 | 战略计划层 | 从战术上升到战略规划 |
| Ch7 | 残局定式库 | 精通终局处理和定式知识 |

这七大支柱共同构成了一个**从底层计算到高层决策**的完整 AI 训练体系。

---

**文档撰写完成标志**：第 1-7 章全部完成 (~50,000 字)

下一步：**实现对应的 Python 算法代码**，并将这些棋理集成到一个 Teacher/Oracle 模型中。

**作者**: Junqi Engine Team  
**版本**: 2.0 (全书完成)  
**最后更新**: 2026-09-05

---

*接下来需要实现算法代码并重构专家模型*
