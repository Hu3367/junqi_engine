# 军棋复盘数据使用指南

## ✅ 完成的工作

已成功批量解析 **1000 局** 军旗移动端 App（libjunqi.so）的二进制复盘存档文件，并生成分析工具链。

---

## 📁 文件清单

### 解析工具
- `batch_parse_replays.py` - 批量解析 .sav 文件的主程序
- `analyze_replays_report.py` - 生成详细统计分析报告
- `query_replays.py` - 交互式查询工具（推荐）

### 输出数据
- `all_replays.json` - 原始完整数据（994 局有效 + 6 局失败）
- `replay_analysis_full.json` - 含开局特征分析的完整版本
- `replay_decisive_games.json` - 仅已结束胜负局（47 局）
- `replay_undecided_games.json` - 未完成对局（943 局）

### 分析报告
- `REPLAY_ANALYSIS_SUMMARY.txt` - 文本格式快速概览
- `REPLAY_ANALYSIS_GUIDE.md` - 详细使用指南
- `TEMP_ANALYSIS.JSON` - 单局复盘 JSON 示例

---

## 🚀 快速开始

### 1. 查看基础统计
```bash
python query_replays.py
```

### 2. 按玩家名查询
```bash
# 查找某玩家的所有对局
python query_replays.py --by-player "棋手 62240"

# 查看某玩家的胜率统计
python query_replays.py --stats "棋手 62240"
```

### 3. 分析开局模式
```bash
python query_replays.py --openers
```

### 4. 查看特定对局详情
```bash
python query_replays.py --detail "251122223849 棋手 62240-棋手 44374.sav"
```

### 5. 只看已结束对局
```bash
python query_replays.py --decided
```

---

## 📊 关键发现

### 数据集统计
| 指标 | 数值 |
|------|------|
| 总复盘数 | 1000 |
| 成功率 | 99.4% (994/1000) |
| 平均着法数 | 110 步 |
| 最长对局 | 456 步 |

### 胜负分布
- **已结束**: 47 局 (先手胜率 53.2%)
- **未完成**: 943 局 (94.9% - 进行中或中途退出)
- **和棋**: 4 局

### 对局模式
- 模式 1 (天梯): 355 局
- 模式 2: 339 局
- 模式 3: 300 局

---

## 🔬 应用场景

### P0 - 正确性验证
```python
# 验证引擎回放合法性
import json
data = json.load(open('all_replays.json'))
ok_count = sum(1 for g in data['games'] if g['result']['replay_ok'])
print('回放成功率:', ok_count, '/', len(data['games']))  # 994/1000 ✓
```

### P1-P3 - AI 训练数据

#### 行为克隆 (P2)
```python
# 使用未完成对局进行 BC 训练
import json
undecided = json.load(open('replay_undecided_games.json'))
# 每局的 final_state 和 moves 可用于学习人类策略
```

#### Value Head 训练 (P3)
```python
# 使用已结束的胜负局训练估值网络
decisive = json.load(open('replay_decisive_games.json'))
# label = winner (0=先手胜，1=后手胜)
# features = 从 final_state 提取棋盘状态
```

### P4 - 基准评估
将 AI 模型的胜率与人类对战数据进行对比：
- 人类先手胜率：53.2%
- 人类后手胜率：46.8%

---

## 💡 数据分析示例

### 示例 1: 查找最强玩家
```bash
python query_replays.py --by-player "棋手 62240" > p1_matches.txt
python query_replays.py --stats "棋手 62240"
```

### 示例 2: 研究长局战术
筛选超过 200 步的对局（共 84 局），分析持久战策略：
```python
import json
data = json.load(open('all_replays.json'))
long_games = [g for g in data['games'] if g['game_info']['total_moves'] > 200]
for g in sorted(long_games, key=lambda x: -x['game_info']['total_moves'])[:5]:
    print(g['filename'], g['game_info']['total_moves'], '步')
```

### 示例 3: 开局偏好分析
观察前两步翻牌组合，识别主流开局：
```bash
python query_replays.py --openers
```

---

## ⚠️ 注意事项

### 数据质量
- ✅ 99.4% 解析成功率
- ❌ 6 个文件解析失败（需手动检查文件格式）

### 数据特点
- ⚠️ 94.9% 为未完成对局 → 适合 Policy Learning，不适合直接回归 Value
- ✅ 47 局已结束对局 → 适合 Value Head 训练和胜负预测

### 时间戳处理
`timestamp_approx` 是近似秒级时间戳（实际精度×256），如需精确时间需重新解码：
```python
# 原始时间戳计算
raw_ts = struct.unpack_from("<I", data[2:5])[0] * 256
# 转换为 Unix 时间
dt = datetime.fromtimestamp(raw_ts)
```

---

## 🔧 自定义分析

修改 `query_replays.py` 添加新功能：

```python
# 按等级分范围过滤
def filter_by_rating(games, min_rating, max_rating):
    return [g for g in games 
            if min_rating <= g['players']['player1']['rating'] <= max_rating]

# 按对局时长过滤
def filter_by_duration(games, min_steps, max_steps):
    return [g for g in games 
            if min_steps <= g['game_info']['total_moves'] <= max_steps]
```

---

## 📞 技术支持

如有问题或需要更多功能：
1. 查看 `REPLAY_ANALYSIS_GUIDE.md` 获取详细 API 文档
2. 直接使用 `parse_sav_replay.py` 分析单局文件
3. 修改上述脚本添加定制化统计

---

## 🎯 下一步建议

1. **建立标签数据集**: 从 47 局已结束对局中抽取关键节点
2. **构建 baselines**: 用人类走法训练一个简单的 Policy Network
3. **特征工程**: 从 `final_state` 提取棋盘特征（兵力对比、军旗位置等）
4. **长期计划**: 持续收集更多高质量复盘数据扩充数据集
