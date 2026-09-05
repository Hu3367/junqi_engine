# Conversation Title: 这个深度学习的计划方案是否具有可行性
Original URL: https://chatgpt.com/c/6a9409b2-e454-83ec-b149-9f052aebcec9
Platform: ChatGPT
Created At: 2026-08-30 18:45:09
Total Messages: 4

---

User：[2026-08-30 18:45:09]
这个深度学习的计划方案是否具有可行性

AI：[2026-08-31 22:57:54]
对，而且我现在反而比较认同你的判断：你这个项目不应该把“自训练”当成主路线。

你描述的“初始大量未知 → 翻开逐渐减少灰雾 → 根据已知信息推断未知 → 边探索边决策”的游戏类型，在博弈 AI 里有一个非常成熟的方向：不完全信息博弈（Imperfect-Information Game）。而你的“军棋-翻棋”，从算法问题上看，和 Chinese Dark Chess / 暗棋 几乎是同一类问题，甚至比 Stratego 更接近。已有针对 Chinese Dark Chess 的程序，已经通过传统搜索 + 概率/随机信息处理做到非常强的竞技水平。
CiNii
+1

所以，你昨天训练出来“不进步、两边互相循环”的结果，其实很有价值：它说明当前这个问题未必适合直接套 AlphaZero。

一、和你最像的，其实就是 Chinese Dark Chess

Chinese Dark Chess 本身就是典型的不完全信息棋类：

棋子初始身份未知；

翻开后逐渐获得信息；

吃子会暴露双方棋力关系；

当前决策依赖于对隐藏棋子的概率判断；

棋盘上的信息量随着对局推进不断增加。

针对它，早期就有人专门研究：

如何把 MCTS 修改成适用于不完全信息的版本。

2011 年的一项 Chinese Dark Chess 研究提出了带随机节点的 MCTS 方法，并报告在当年的电脑暗棋竞赛中取得非常强的成绩；后续研究又继续加入 chance nodes、quality-based rewards、implicit minimax backups、progressive bias 等技术。
CiNii
+1

尤其值得注意的是，2016 年对 DarkKnight 的研究里，单独加入不同增强技术后，胜率提升幅度很明显，其中 progressive bias 的效果尤其突出，综合使用这些技术后，对原版本的胜率达到 84.75%。
科学直达

这对你的项目非常重要，因为它说明：

不需要先训练一个神经网络，传统程序 + 概率推理 + 搜索，就完全可以把这种游戏做得很强。

二、甚至 Stratego 也是同类问题，但比你的问题更“重”

另外一个著名案例就是 Stratego。

它和军旗有非常明显的共同点：

棋子身份隐藏
       ↓
通过交战获得信息
       ↓
不断更新对敌方棋子的判断
       ↓
在不确定信息下决策

DeepMind 的 DeepNash 最终把 Stratego 做到了非常高水平，但它走的是一个很重的方向：博弈论 + 深度强化学习 + Nash Dynamics，而不是 AlphaZero 式的传统 MCTS。DeepMind 自己也明确指出，Stratego 的不完全信息和巨大状态空间使传统 game-tree search 很难直接扩展。
Google DeepMind

所以这里其实有两个极端：

传统程序路线
Chinese Dark Chess
       ↑
       │
       │ 你现在应该首先走这里
       ↓
Belief + Heuristic + Search

深度学习路线
Stratego / DeepNash
       ↓
复杂 RL / Game Theory

你昨天做的东西，是直接从：

AlphaZero

跳到了：

DeepNash 类问题

对于目前阶段来说，我认为跨度太大。

三、你说的“灰雾逐步探索”，其实有一个非常漂亮的传统算法模型

我建议你把 AI 的核心思想重新定义成：

“我不是要猜出对手是什么，而是维护一组对对手是什么的概率假设，并根据每一步的新信息不断缩小这个集合。”

也就是：

Belief State

例如某个敌方暗子：

位置 A

司令  5%
军长 15%
师长 30%
旅长 25%
团长 15%
营长 10%

此时 AI 不需要说：

“它就是师长。”

而应该说：

“我认为它是师长的概率最高。”

然后下一次发生：

这个棋子攻击了一个我方棋子
结果：
敌方胜

那么立刻更新：

司令   2%
军长   5%
师长   18%
旅长  35%
团长  25%
营长  15%

整个系统就是不断进行：

Belief
t+1
	​

=Update(Belief
t
	​

,Observation
t
	​

)

这其实比“让神经网络自己悟出这些概率”容易得多。

四、因此我非常建议你把 AI 拆成 5 个传统模块

我认为这是你目前最有希望快速做出普通人水平 AI 的方案。

              ┌───────────────┐
              │   Game State   │
              └───────┬───────┘
                      ↓
             ┌─────────────────┐
             │  Belief Engine  │
             │ 敌方暗子概率模型 │
             └───────┬─────────┘
                     ↓
          ┌──────────────────────┐
          │ Candidate Move Gen   │
          │ 候选走法生成          │
          └──────────┬───────────┘
                     ↓
          ┌──────────────────────┐
          │ Imperfect Search     │
          │ 多世界搜索 / MCTS    │
          └──────────┬───────────┘
                     ↓
          ┌──────────────────────┐
          │ Heuristic Evaluation │
          │ 局面评价              │
          └──────────┬───────────┘
                     ↓
                 Best Move

你现在项目里已经有：

rules.py
state.py
ai.py
calculator.py

那么其实非常适合在这套基础上直接演进，而不是推倒重来。

五、其中最关键的不是 MCTS，而是“概率推理器”

这个东西我认为甚至应该独立成为：

belief.py

维护：

Python
运行
belief[piece_position][piece_type]

例如：

belief[12] = {
    COMMANDER: 0.03,
    GENERAL:   0.11,
    COLONEL:   0.22,
    MAJOR:     0.27,
    ...
}

然后每发生一个事件：

翻棋
吃子
被吃
胜负关系暴露
某棋子移动
某类型棋子已经全部出现

就重新计算。

而且这里不一定需要机器学习。

完全可以：

Bayesian Update
+
棋子数量约束
+
规则约束
+
历史行为约束

甚至可以进一步做：

Opponent Modeling

例如某个对手连续表现：

不敢攻击
喜欢保护高价值棋
喜欢主动探子
喜欢用小子试探

AI 就逐渐知道：

“这个玩家的行为风格是什么。”

这已经非常接近人类思考方式了。

六、然后才是“灰雾探索”的核心：信息价值

这恰恰是你当前 AI 很可能完全没有体现出来的东西。

例如：

行动 A：吃一个已知低级棋子
行动 B：翻开一个危险位置的暗子
行动 C：移动高价值棋

传统“吃子评分”可能认为：

A = +10
B = 0
C = +2

所以选 A。

但是一个优秀人类玩家会想：

B 虽然当前没有吃子收益，但是它可以大幅降低我的不确定性。

这就是：

Information Gain

可以定义：

IG(a)=H(Belief
before
	​

)−E[H(Belief
after
	​

∣a)]

简单讲就是：

走完这一步，我对棋局知道得更多了吗？

于是：

吃掉普通棋子
Material Gain = +5
Information Gain = +1

翻一个关键暗子
Material Gain = 0
Information Gain = +15

后者可能反而是更好的动作。

七、这就是为什么“翻棋”很适合传统程序

你的游戏其实有一个非常漂亮的特点：

信息本身就是资源。

所以评估函数完全可以变成：

Evaluation=Material+Position+Mobility+Safety+Information−Risk

例如：

Material       40%
Position       20%
Mobility       10%
Safety         15%
Information    10%
Risk            5%

这些权重甚至一开始都可以手工设定。

然后再通过大量对局慢慢调。

这远比：

随机 ResNet
↓
随机 self-play
↓
希望它悟出来

容易控制。

八、MCTS 仍然可以用，但我建议不要用 AlphaZero 那套 MCTS

这一点很关键。

你现在做的是：

NN Policy
   ↓
PUCT
   ↓
NN Value

我建议改成：

Belief State
    ↓
Sample World 1
Sample World 2
Sample World 3
...
    ↓
Search
    ↓
统计各动作在不同可能世界中的表现

也就是：

多世界搜索 / belief-aware search

例如：

当前暗子有三种可能：

W1：军长
W2：师长
W3：炸弹

动作 A：
W1 +8
W2 +7
W3 -20

动作 B：
W1 +4
W2 +5
W3 +3

那么 AI 会发现：

A 平均收益很好
但风险极高

B 平均稍低
但非常稳定

如果你的 AI 想模拟普通人的水平，实际上这种：

“收益 × 风险 × 信息”

就是非常合理的决策方式。

九、而且这个方案特别适合解决你昨天的“循环棋”

因为传统程序完全可以直接规定：

如果最近出现：

A → B → A → B

那么：

repeat_count += 1

然后搜索评价：

Eval
′
=Eval−λ⋅Repeat

同时：

无吃子超过 N 手
       ↓
增加 stagnation penalty

甚至可以直接规定：

在多个局面价值近似时，优先选择能够改变局面的动作。

这属于搜索策略设计，不需要 AI 自己“学会”。

十、还有一个非常适合你的技巧：Progressive Bias

这个正好和你之前的问题高度对应。

研究 Chinese Dark Chess 的 DarkKnight 时发现，Progressive Bias 对棋力提升非常明显。
科学直达

简单理解就是：

MCTS 初期不要完全相信随机统计，而是给“人类认为合理的动作”一些先验偏置。

例如：

吃高价值棋
+
控制关键位置
+
减少自身暴露
+
主动获取信息
+
避免重复

先给这些动作一点 bias。

随着搜索次数增加：

经验统计
     ↓
逐渐压过
     ↓
人工 bias

于是：

搜索少的时候
→ 有人类常识兜底

搜索多的时候
→ 算法逐渐自己判断

这非常适合你现在这个阶段。

十一、所以我认为“普通人水平”完全不需要深度学习

这个判断我是比较有把握的。

甚至可以把难度分成：

Level 1：弱 AI
固定棋子价值
+
合法走法
+
简单吃子优先
+
随机探索

已经不像纯随机棋。

Level 2：普通玩家

加入：

Belief Model
+
棋子数量推断
+
简单搜索
+
防止送高价值棋
+
避免循环
+
基本位置价值

我认为这一级完全可以通过传统程序做到。

Level 3：较强玩家

再加入：

多世界搜索
+
Opponent Modeling
+
Information Gain
+
Progressive Bias
+
残局搜索

这一阶段可以明显超过纯启发式 AI。

Level 4：高手

才开始考虑：

Neural Value
+
Self-play
+
Deep RL
+
Game-theoretic training

而不是一开始就上。

十二、我甚至建议你的项目战略直接改成“两条腿走路”

不要删掉你现在的深度学习代码。

而是变成：

                    军棋 AI
                       │
           ┌───────────┴────────────┐
           ↓                        ↓
      Classical AI              Neural AI
           │                        │
   Belief + Search           ResNet + RL
           │                        │
           └───────────┬────────────┘
                       ↓
                  Tournament
                       ↓
                  Elo / 胜率

然后让两套 AI 真正竞争。

这样你会得到一个很有价值的实验：

Classic V1
Classic V2
NN V1
NN V2
Hybrid V1

而不是凭感觉判断：

“这个神经网络好像学会了。”

十三、实际上你现在最值得做的不是继续训练，而是做一个“Classic V1”

我会这样设计：

Classic V1
│
├── 1. 完整合法走法生成
│
├── 2. Belief Engine
│      ├─ 棋子身份概率
│      ├─ 剩余棋子数量约束
│      └─ 历史行为更新
│
├── 3. Evaluation
│      ├─ 棋子期望价值
│      ├─ 位置价值
│      ├─ 机动性
│      ├─ 风险
│      ├─ 信息价值
│      └─ 循环惩罚
│
├── 4. Search
│      ├─ 短深度确定性搜索
│      └─ 多世界采样
│
├── 5. Opponent Model
│
└── 6. Endgame Rules
然后先不用 GPU。

甚至：

Python
+
NumPy
+
你的 rules.py

就可以开始。

因为目前真正的瓶颈不是算力，而是决策模型是否正确。

十四、还有一个很有意思的结论

你这个项目其实非常适合采用：

“80% 规则/搜索 + 20% 学习”

而不是：

“100% 神经网络。”

比如：

Belief：
      规则 + Bayesian

合法行动：
      规则

循环：
      规则

棋子基础价值：
      规则

搜索：
      MCTS / Alpha-Beta 类

对手行为：
      简单统计

评价函数参数：
      学习

最终局面 Value：
      Neural Network（以后）

这样做有一个特别大的优势：

AI 的每一个行为你都能解释。

例如你以后发现 AI 为什么没有吃某个棋，可以直接输出：

候选 A
Material     +18
Information   +2
Risk         -25
Loop          0
Total         -5

候选 B
Material      +8
Information  +12
Risk          -2
Loop          0
Total        +18

你就知道它为什么走 B。

而当前神经网络出现：

“AI 就是这么预测的。”

你很难知道为什么。

我的结论

你的判断是对的。

对于“军旗-翻棋”这种灰雾逐步揭示 + 隐藏棋子推理 + 探索与对抗同时进行的游戏，最匹配的研究路线并不是纯 AlphaZero，而是：

Belief State + Bayesian 推理 + 多世界搜索 + 启发式评估 + Opponent Modeling

而且这不是我的臆测，Chinese Dark Chess 已经有相当成熟的传统 AI 研究路线，相关程序甚至在国际电脑棋类赛事中取得过很好的成绩。
CiNii
+2
阳明交通大学学术中心
+2

对于你目前的项目，我会把路线从：

AlphaZero → Self-play → 期望学会军棋

改成：

                游戏规则
                   ↓
             Belief Engine
                   ↓
       ┌───────────┴───────────┐
       ↓                       ↓
   局面评估                 信息评估
       ↓                       ↓
       └──────────┬────────────┘
                  ↓
            Multi-World Search
                  ↓
          Progressive Bias
                  ↓
             Best Action

先用传统算法做出一个“会下棋的 AI”，再把深度学习作为增强器，而不是让深度学习负责从零学习什么叫“会下棋”。

这条路线我认为比你昨天的方案更适合当前项目，也更容易在短时间内达到普通玩家水平。

而且你现有的 rules.py / state.py / ai.py / marginal() 基础已经非常适合直接转成这个架构。

d46601b9-5e45-41f4-b398-989b74f… +1

如果按你的项目继续往下做，我下一步最建议直接把 “Classic AI V1”的完整算法架构和评分函数设计出来，包括 belief.py / evaluator.py / search.py / opponent_model.py 四个核心模块，以及
每一步具体怎么算分、怎么算暗子概率、怎么处理循环和“翻子的信息价值”。

AI：[2026-08-31 22:57:54]
是的，而且我在查了这批资料以后，要修正我上一轮对你项目的一个判断：

AlphaZero 并不是不适合军棋翻棋；相反，Chinese Dark Chess/Banqi 恰好已经有一条被实战和论文验证过的 AlphaZero 路线。

但这里有一个非常关键的细节：Banqi 的“隐藏信息”与扑克、Stratego 那种私有信息并不完全一样。 Chinese Dark Chess 的隐藏棋子对双方是对称的，双方都不知道同一批暗子的身份，因此研究者可以把“翻子结果”建模成随机/机会节点。2015 年的工作就明确这样处理，并用 NMCTS + 启发式方法做到了多次比赛冠军；2018、2023 年又进一步验证了 AlphaZero 在 Chinese Dark Chess 的简化随机版本上可以学到接近最优策略。
ResearchGate
+1

更重要的是：你昨天遇到的“双方不断循环、训练没有进步”，居然已经在 Banqi 的 AlphaZero 研究里被明确观察到了。 2023 年论文直接发现，cpuct 太高会让 MCTS 在 2×4 Chinese Dark Chess 中接近随机行动，而随机行动会增加重复局面，最终因为三次重复而大量和棋。
JAIST Repository

所以你现在真正需要的不是“换不用 AlphaZero”，而是：

按照已经验证过的 Banqi/AlphaZero 研究路线，把训练系统拆成可验证的阶段，而不是直接在 4×8 全尺寸上盲跑 Self-play。

一、先看最重要的结论：AlphaZero 为什么在 Banqi 真的能成立？

2018 年，Hsueh、Wu、Jr-Chang Chen 等人专门做过：

AlphaZero for a Non-deterministic Game

他们不是拿普通 AlphaZero 生搬硬套，而是先把 Chinese Dark Chess 缩小成已经解决的 2×4 CDC，然后用理论最优结果作为“标准答案”。

实验结论非常直接：

在很多超参数设置下，AlphaZero 能收敛到接近理论值和最优策略。
Hsuehch
+1

这件事情对你的意义非常大。

因为它解决了一个你现在最大的开发问题：

以前：
“这个训练结果到底算好还是不好？”
        ↓
没有答案

而他们的办法是：

先做一个小型、可求解的 Banqi
        ↓
理论最优值已知
        ↓
AI训练
        ↓
直接比较

所以不是：

“我觉得 AI 好像进步了。”

而是：

“第 20 轮与理论最优值的误差从 0.18 降到 0.03。”

这就是可科学迭代的训练。

二、实际上你昨天的问题，在论文里几乎原样出现了

这个地方非常值得你注意。

2023 年对多个随机博弈变体的 AlphaZero 分析中，他们发现：

对 2×4 CDC，过高的 cpuct 会使 MCTS 的动作选择接近随机；随机下棋会增加局面重复，并因为重复规则导致大量和棋。
JAIST Repository

也就是说：

你昨天看到的：

AI
↓
没有明显策略
↓
疯狂探索
↓
走回头棋
↓
重复
↓
和棋
↓
训练不进步

并不一定是“奖励没有吃子分”造成的。

其中一个非常现实的原因可能是：

MCTS探索强度
        ↑
      太高
        ↓
Self-play质量下降
        ↓
大量随机/循环
        ↓
Replay Buffer被垃圾数据污染

这比“给吃子 +1”更接近你当前问题的本质。

三、真正让我觉得对你项目特别有价值的是：他们已经把“循环信息”输入网络了

你现在的方案是 29 个输入平面。

而 2023 年的 CLAP 实验中，CDC 的 DNN 输入是 33 个平面，其中明确包含：

棋子状态的 one-hot 编码；

每种类型的未翻开棋子数量；

重复次数信息；

当前轮到谁；

距离“无翻子/无吃子和棋”阈值的进度信息。

然后使用 5-block ResNet + Policy/Value 双头。
JAIST Repository

这点与你现在的设计几乎一一对应。

你目前方案里已经有：

无吃子步数归一化

但没有看到明确的：

repetition count

而这是 CLAP 的实际输入设计之一。
JAIST Repository

所以我会建议你立刻把：

29 planes

改成至少：

棋子信息
+
未翻子统计
+
side to move
+
repetition count
+
draw-progress

而且这里不是我凭经验建议，是有 Banqi AlphaZero 的实际论文实现可以参考。

四、还有一个非常重要的地方：他们不是只有一个简单的 Value

你当前方案：

Value Head
→ 一个 V ∈ [-1,1]

而 CLAP 的实验配置中：

Value Head
→ 2-dimensional
→ 估计两个玩家的 expected outcomes

JAIST Repository

这非常值得你考虑。

因为在翻棋这种随机游戏里：

我的胜率
+
对手胜率
+
和棋概率

本身就是非常有价值的信息。

例如：

P(win)  = 0.63
P(draw) = 0.22
P(loss) = 0.15

比一个：

V = +0.48

的信息量大得多。

你甚至可以把 Value Head 设计成：

V(s)=[P
win
	​

,P
draw
	​

,P
loss
	​

]

然后保证：

P
win
	​

+P
draw
	​

+P
loss
	​

=1

当然，这已经是我的设计建议，不是论文原始实现；论文里明确的是双维 expected outcomes。
JAIST Repository

对于你昨天遇到的“和棋污染训练”，这种表示尤其值得尝试。

五、而且 CLAP 的训练规模其实给了你一个非常有价值的参考答案

2023 年论文的 CLAP 实验：

5-block ResNet
        ↓
每轮 1000 局 Self-play
        ↓
总共 50 iterations
        ↓
Replay Buffer = 40,000
        ↓
Batch = 256
        ↓
SGD + Momentum 0.9
        ↓
Weight decay 0.0001
        ↓
Nesterov
        ↓
LR 0.1
25轮后 → 0.01

JAIST Repository

这与你现在方案里的：

“近数万手 replay buffer + self-play + 训练”

实际上非常接近。

d46601b9-5e45-41f4-b398-989b74f…

所以你的工程思路本身没有错。

真正的问题是：

你现在是在“没有经过校准的完整 4×8 游戏”上验证整个体系。

而论文路线是：

2×4 / 小规模
↓
理论最优验证
↓
超参数验证
↓
DNN验证
↓
再逐渐推广
六、2018 年那篇论文甚至把“应该怎么调参”研究得很详细

这是我目前最推荐你认真读的一篇。

他们在 2×4 CDC 上系统测试了：

cpuct
Dirichlet α
Dirichlet ε
Temperature τ

其中基准设置：

cpuct = 1
Dirichlet α = 1.5
Dirichlet ε = 0.25
Temperature = 1

自对弈：

1000 games / iteration
50 iterations
800 simulations / move

然后逐项改变参数。
Hsuehch
+1

有一个结论与你现在特别相关：

cpuct 太低学不出来；但太高也会让学习速度下降甚至产生不稳定。
Hsuehch

而且：

太高的探索并不一定等于更好的学习，因为可能只是产生更多随机局面。
Hsuehch

所以以后你不要再凭感觉调：

sims 20
→ 50
→ 100
→ 200
→ 500

然后看哪个“感觉更好”。

应该建立实验矩阵。

七、最重要的一个认识：不要把“训练轮数”作为主要进步指标

这是你现在最容易陷入的坑。

你可能会看到：

Iteration 1
Iteration 2
Iteration 3
...
Iteration 50

然后期待：

AI越来越聪明

但 AlphaZero 本身并不是保证“每轮都变强”。

2023 年论文明确做了不同随机种子的重复实验，发现默认参数的不同 trial 之间总体表现具有较好的重复性，但它们仍然需要通过明确指标来判断策略和值是否学得正确。
JAIST Repository

所以你应该至少建立三个独立指标：

① Policy Quality

让模型打固定基准：

固定强 AI
固定随机 AI
历史最强模型

测胜率。

② Value Quality

对已经知道理论答案的残局：

Predicted value
vs
Tablebase value

算 MAE。

③ Exploration Quality

统计：

Distinct states
Repeat rate
Draw rate
Average game length

这三个指标结合起来，你才知道：

是搜索变强了？
还是只是更加随机？

2018/2023 的研究就是这么做的。
Hsuehch
+1

八、我认为你现在最应该建立一个“Banqi AI 实验靶场”

这会彻底解决你说的：

“像盲人摸象一样尝试。”

我建议直接把项目分成四个难度级别。

Level 0：极小型 Banqi

例如：

2×4
+
8 pieces

目标不是训练出实战 AI。

而是：

验证 Encoder
验证 MCTS
验证 Policy
验证 Value
验证循环机制
验证奖励
验证超参数

甚至可以直接拿公开论文的理论结果做对照。

Level 1：小规模可求解 Banqi

逐渐增加：

棋盘
+
棋子数量
+
隐藏信息

但仍然能够通过残局数据库或穷举获得准确答案。

然后：

AI
vs
Tablebase

每一次代码改动都能测：

+2.3%
-0.7%

这样你第一次拥有真正的：

回归测试。

九、Level 2：完整 4×8，但先不要追求“自我进化”

这里开始使用你真正的军棋。

但是建立固定基准：

Random
Greedy
Search1
Search2
Search3

然后：

NN-v0
NN-v1
NN-v2
NN-v3

全部互相对战。

这样你就有：

Elo Ladder

例如：

Random        1000
Greedy        1180
Search2       1320
Search3       1450
NN-v1         1200
NN-v2         1290
NN-v3         1390

这时你才知道：

到底有没有真的进步。

十、Level 3 才是你真正的 AlphaZero Self-play

这时候才：

最佳模型
   ↓
Self-play
   ↓
Replay Buffer
   ↓
训练新模型
   ↓
vs Best Model
   ↓
胜率达标
   ↓
晋级

我反而建议你不要完全照搬“单模型不断滚动更新”。

自己做一个：

Champion / Challenger
Champion
    ↑
    │ 通过测试
    │
Challenger
    ↑
    │
 Training

例如：

新模型 vs 当前冠军
1000 局

胜率 > 52%
并且置信区间满足要求
        ↓
晋级

否则：

丢弃

这个机制是我的工程建议，但思想与早期 AlphaGo Zero 的 best-model evaluation 很接近；AlphaGo Zero 会让新网络与当前最佳网络对战，达到指定胜率才替换。
Hsuehch

这对你尤其重要，因为它能防止：

“某一次训练把 AI 训练坏了，下一轮继续拿坏模型训练。”

十一、其实早期的 Banqi 强 AI 还有一个你现在非常值得借鉴的东西：不要什么都交给网络

2015 年 Diablo 的路线非常值得你看。

他们做的是：

MCTS
+
Random / Chance Nodes
+
Flipping Heuristics
+
Simulation Policy
+
Shorter Simulation

而不是纯神经网络。
ResearchGate

尤其他们专门设计了：

翻棋启发式

根据暗子周围已经翻开的棋子，对翻子动作进行打分。
ResearchGate

同时还专门处理：

无效走子 / 重复走子导致的和棋问题。 
ResearchGate

这个思路跟你现在的问题高度吻合。

十二、还有一个非常重要的研究方向：残局数据库

这一条其实更适合你现在。

2024 年还有一篇专门研究：

Improvements in Evaluation Functions Based on Endgame Databases of Chinese Dark Chess

其做法是利用 6-man endgame database，从已经解决的残局中提取不同棋子组合的胜负比例，然后构造更大棋子数下的 Material Combination Values，再把它融入评价函数。
EurekaMag
+1

这意味着：

不是所有东西都必须通过 RL 学出来。

你完全可以：

残局：
Tablebase / Exact Solver

中局：
MCTS + NN

开局：
NN + MCTS + Chance

这其实是一个非常成熟的工程思路。

十三、这也回答了你最开始的疑问：为什么 CLAP_CDC 可以用 AlphaZero？

因为 Banqi 与 Stratego 的难点虽然相似，但信息结构不完全一样。

Stratego
我不知道你的棋子
你也不知道我的棋子

但：
你知道你的私有信息
我不知道

属于真正的 asymmetric / private information。

因此会遇到：

Opponent modeling
Mixed strategy
Game theory
Nash equilibrium

这也是 DeepNash 那种路线出现的原因。

Chinese Dark Chess

则更特殊：

暗棋 A 是什么？

我不知道
你也不知道

而且双方观察到的翻子结果完全一致。

因此可以把：

未知身份
↓
随机变量
↓
Chance Node

处理。

2015 年的论文就明确指出，Chinese Dark Chess 的隐藏信息是对称的，而且暗子类型的 belief distribution 可以直接计算；翻子是唯一引入随机性的动作。
ResearchGate

这就是为什么：

AlphaZero/MCTS 在 Banqi 上比在 Stratego 上更自然。

十四、但是要特别注意一个“看起来矛盾”的地方

2023 年那篇 AlphaZero 论文又写了一句很重要的话：

对 poker 等真正的不完全信息游戏，可能需要 opponent modeling / mixed strategies；AlphaZero 是否适用于这类 imperfect-information games 不在该论文研究范围内。
ResearchGate

这并不矛盾。

他们研究里的 CDC 被作为：

stochastic game

来处理，而不是把它当成 poker 那种私有信息博弈。

所以你现在的 AI：

Belief
+
Chance
+
MCTS
+
AlphaZero

是有理论基础的。

但不要把它简单理解成：

“AlphaZero 已经解决了所有迷雾博弈。”

它解决的是更适合这种对称隐藏信息 + 随机揭示结构的情况。

十五、回到你自己的项目，我现在会建议你彻底调整开发路线

你之前的方案是：

ResNet
↓
MCTS
↓
Self-play
↓
Replay
↓
不断训练

我建议改成：

                    ┌──────────────┐
                    │  Rule Engine │
                    └──────┬───────┘
                           ↓
                  ┌─────────────────┐
                  │  Observation    │
                  │  + Chance Model │
                  └────────┬────────┘
                           ↓
                    ┌────────────┐
                    │   Encoder  │
                    └─────┬──────┘
                          ↓
                  ┌──────────────┐
                  │  ResNet       │
                  │ Policy+Value  │
                  └──────┬───────┘
                         ↓
                  ┌──────────────┐
                  │ Chance-aware  │
                  │     MCTS      │
                  └──────┬───────┘
                         ↓
                     Action
                         │
                         ↓
                   Self-play
                         │
                         ↓
                  ┌──────────────┐
                  │ Replay Buffer │
                  └──────┬───────┘
                         ↓
                      Training
                         ↓
                  ┌──────────────┐
                  │ Challenger NN │
                  └──────┬───────┘
                         ↓
                  Tournament Test
                     ↙       ↘
                Champion   Tablebase/
                /Baseline   Strong AI
十六、而我认为你现在最应该做的是这个“4阶段路线”
阶段 A：验证算法，而不是验证棋力

用小型 Banqi：

2×4

目标：

Value MAE ↓
Policy error ↓
Win rate ↑

达到论文级别结果后，再继续。

阶段 B：验证网络

用论文里的结构做一个基准：

5-block ResNet
33 planes
Policy head
2-value head
Replay = 40k
Batch = 256

这时候不要自己发明很多东西。

先尽量复现公开研究结果。

JAIST Repository

阶段 C：4×8 CDC

先：

NN + MCTS

不要急着训练。

让它打：

random
greedy
search2
search3

把：

胜率
和棋率
平均手数
重复率

全部记录下来。

阶段 D：Self-play

最后再：

Self-play
+
Replay
+
Champion / Challenger
+
Elo

这时候你每次训练都能说：

“V7 比 V6 高 37 Elo。”

而不是：

“V7 好像比 V6 更聪明。”

十七、还有一个很重要的事实：CLAP_CDC 并不是“2024 年才突然成功”

这也是我查资料以后觉得非常有意思的一点。

同一团队的：

2021
CLAP_CDC
↓
2022
CLAP_CDC 金牌
↓
2023
CLAP_CDC 银牌
↓
2024
CLAP_CDC 金牌

研究者 Chu-Hsuan Hsueh 的个人学术履历明确记录了这些成绩；2024 年 CLAP_CDC 再次获得第 27 届 Computer Olympiad Chinese Dark Chess 金牌。
ResearchMap

所以这不是：

“他们把 AlphaZero 跑了一次，刚好赢了。”

而是一个持续多年迭代的程序。

目前公开的 2024 年比赛报告只明确说明六支队伍参赛、CLAP_CDC 使用 AlphaZero 并夺冠，并且讨论了两盘对局；这篇 2024 比赛报告全文目前在 Sage 上属于受限访问，我没有找到足够可靠的公开全文来逐队核实六支队伍的最新内部训练配方，因此这里不能把各队 2024 年的具体实现细节说得过头。
Sage Journals
+1

但我们已经能从这些队伍公开发表的技术路线中看到一条非常清晰的演化：

2010
Alpha-Beta + Chance

        ↓

2011~2016
NMCTS + Heuristics
+ Early termination
+ Quality rewards
+ Progressive bias

        ↓

2018
AlphaZero on solved 2×4 CDC

        ↓

2023
Tabular AlphaZero
+ DNN
+ CLAP
+ 4×8 CDC variants
+ systematic hyperparameter study

        ↓

2024
CLAP_CDC
→ Computer Olympiad Champion

Sage Journals
+4
ResearchGate
+4
科学直达
+4

十八、所以我现在给你的核心建议只有一句

不要再从“怎么让 4×8 AlphaZero 自己学会军棋”开始。

应该从：

“怎么建立一个能够告诉我每一次改动到底变强还是变弱的实验体系”开始。

然后严格按：

可求解小 Banqi
       ↓
复现论文 AlphaZero
       ↓
验证 MCTS 超参数
       ↓
验证 DNN
       ↓
4×8 Baseline
       ↓
Champion / Challenger
       ↓
Self-play
       ↓
持续迭代

这条路线的最大价值不是让你一次做出最强 AI，而是让你从此摆脱：

“改一个参数 → 跑几千局 → 看起来好像没变 → 再改一个参数”

这种盲人摸象式开发。

我认为你下一步甚至应该暂时停止训练

先把你的当前工程改成一个**“Banqi AI Benchmark”**：

metrics/
├── win_rate.json
├── draw_rate.json
├── repeat_rate.json
├── avg_game_length.json
├── policy_loss.json
├── value_mae.json
├── elo.json
└── training_curve.json

然后做：

V0
→ V1
→ V2
→ V3

每个版本都必须回答：

棋力 ↑ 还是 ↓？
和棋率 ↑ 还是 ↓？
循环率 ↑ 还是 ↓？
Value 是否更准？
Policy 是否更稳定？

这套基础设施搭好以后，你再决定 AlphaZero、传统 MCTS、启发式或者混合方案哪个值得继续投资，就会非常清楚。

而从目前查到的资料来看，我会把你的项目主路线定为：

Chance-aware AlphaZero / Neural MCTS，而不是放弃 AlphaZero；但用“小型可求解 → 可验证 → 再扩展到 4×8”的研究路线来推进。

最值得你马上看的三份资料

第一篇是 2018 年的 AlphaZero for a Non-Deterministic Game，它最适合回答“AlphaZero 到底能不能用于 Banqi”。
Hsuehch

第二篇是 2023 年的 Analyses of Tabular AlphaZero on Strongly-Solved Stochastic Games，这是我认为对你目前开发价值最大的资料，因为里面直接给了 CDC 的输入平面、ResNet、Replay Buffer、Batch、训练轮数，以及 cpuct/Dirichlet/Nsim 对训练的影响。
JAIST Repository
+1

第三篇是 2016 年的 An analysis for strength improvement of an MCTS-based program playing Chinese dark chess，它告诉你即使不依赖深度学习，Banqi 的 MCTS 也有一套经过实测有效的增强手段：early playout termination、implicit minimax backup、quality-based reward、progressive bias，其中 progressive bias 的单项提升尤其显著，四者结合后的对战胜率达到 84.75%（相对原版 DarkKnight）。
科学直达

我现在非常建议你把你目前的 encoder.py / net.py / mcts.py / train_rl.py 实际代码或项目压缩包给我。按照这些论文中的已验证路线，我可以直接帮你做一次“论文对照审计”，逐项指出你目前和 CLAP/论文方案到底差在哪里，再给你定一个 V1 → V2 → V3 的可量化训练路线，这样会比继续盲目跑 self-play 有价值得多。

AI：[2026-08-31 22:57:54]
从目前能查到的公开资料看，还不能下结论说“顶尖 AI 已经像 Stockfish 对象棋那样把顶尖暗棋人类完全碾压”。更准确的判断是：

顶级 Chinese Dark Chess AI 已经明显超过普通人水平，并且已经在计算机比赛中形成非常强的程序生态；但公开资料里缺乏足够可信的“顶尖人类棋手 vs 顶级 AI”大样本对战，因此无法给出一个类似“AI 对顶尖人类胜率 85%/95%”这样的可靠数字。

这和中国象棋的情况有明显区别。

1. 最直接的人类实战数据：AI 能达到“普通人水平”

最早比较完整的一组数据来自 2015 年 Diablo 论文。

他们让改进版 Diablo 在一个真实暗棋网站上随机匹配 658 名人类玩家。当时：

人类每步允许思考 15 秒

Diablo 每步仅 1 秒

Diablo 每步约 10,000 次模拟

由于网站随机匹配，作者认为这 658 人的平均水平接近网站的普通玩家水平。
ResearchGate

论文没有把这里的“普通人”解释成高手或顶尖玩家，而且作者明确说这个版本的 Diablo 为了作为商业应用与人类互动，进行了削弱，包括禁止重复走法、降低搜索量等，因此不能拿这组实验评价顶尖 AI 对顶尖人类的水平差距。论文最终只得出“能够与普通人竞争”的结论。
ResearchGate

所以这一数据可以回答：

2014～2015 年的强传统 AI 已经能打普通人。

但不能回答：

它能不能打赢暗棋高手。

2. 还有一个很有意思的数据：六层 Alpha-Beta 已经能达到“正常人水平”

我找到一份台湾的暗棋 AI 实验报告，对 4×8 Chinese Dark Chess 做了真人对战。

他们测试了：

Random

Alpha-Beta 2 层

Alpha-Beta 4 层

Alpha-Beta 6 层

商业暗棋软件

真人

其中六层 Alpha-Beta 对“有玩过暗棋”的真人，在记录的实验中胜率约 60%；对“不会玩的真人”约 100%。样本并不大，而且只有少数真人，所以不能把 60% 当成严谨 rating 数据，但它至少再次说明：

传统搜索算法在这个游戏上并不需要深度学习就能达到普通玩家附近甚至以上的水平。 
mxeduc
+1

3. 真正强的程序，已经远远超过“普通 AI”这个级别

这才是你现在最值得关注的地方。

2011～2013 年的 Diablo / DarkKnight 已经在 Computer Olympiad 上连续取得非常强的成绩。

例如 Diablo 在五次计算机暗棋比赛里，使用 500,000 simulations/move，只输了 8 局；同期 DarkKnight 又拿下了 2013 TCGA 和 Computer Olympiad 的冠军。
ResearchGate

而且 2016 年对 DarkKnight 的研究进一步证明，仅仅通过：

Early Playout Termination

Implicit Minimax Backup

Quality-based Reward

Progressive Bias

这些 MCTS 技术改进，就可以让增强版对原版的胜率达到 60.75%、71.85%、59.00%、82.10%；全部结合时达到 84.75%。
科学直达
+1

这说明一个特别重要的事实：

Chinese Dark Chess 的棋力提升空间非常大，而且很大一部分可以来自搜索与评价机制本身，而不是单纯靠更大的神经网络。

4. 到 2024 年，情况已经发生变化：AlphaZero 程序拿了冠军

2024 Computer Olympiad Chinese Dark Chess 有 6 支队伍参赛，冠军是 CLAP_CDC，而且公开报告明确称它是 AlphaZero-based program。
Sage Journals
+1

更值得注意的是，这不是“一次偶然夺冠”。

CLAP_CDC 的团队从 2022 年开始就连续在这个项目中拿到高名次，2022 金牌、2023 银牌、2024 金牌；DarkKnight 同时也长期保持竞争力。
ResearchMap
+1

因此目前可以比较确定地说：

普通人
    ↓
传统启发式 AI
    ↓
强化 MCTS / DarkKnight / Diablo
    ↓
AlphaZero / CLAP_CDC

程序棋力已经发展得非常强。

5. 但是：为什么我仍然不愿意说“AI 已经碾压顶尖人类”？

因为缺少最关键的一组数据：

顶尖人类暗棋选手 vs 顶级 AI，在统一时间限制和足够多局数下的正式对战。

我查到的公开研究，绝大多数评价方式是：

AI vs AI
AI vs benchmark program
AI vs 普通随机人类
AI vs 历届 Computer Olympiad 程序

而不是：

CLAP_CDC
vs
台湾/中国顶尖暗棋高手
1000局

2024 年比赛报告本身也主要是在讲 6 支程序队伍之间的比赛以及具体对局，并没有给出“顶尖人类对战 CLAP_CDC”的统计。
Sage Journals

因此，如果有人现在直接说：

“2024 年 CLAP_CDC 已经 90% 胜率碾压世界顶尖暗棋高手。”

我目前没有找到可靠公开证据支持这个数字。

6. 这和中国象棋其实差别很大

中国象棋已经形成了成熟的：

顶尖职业棋手
vs
顶尖引擎

生态，可以讨论：

让几子
Elo 差
胜率
比赛积分

而 Chinese Dark Chess 没有这么成熟的人类职业竞技体系。

这并不意味着暗棋 AI 不强。

恰恰相反：

暗棋 AI 的“机器顶尖水平”已经非常高，但“机器顶尖水平与人类顶尖水平之间的距离”没有像中国象棋那样被系统测量。

这是两个不同的问题。

7. 我反而觉得暗棋存在一个很有意思的情况：AI 可能没有“绝对碾压”那么简单

因为它是随机 + 隐藏信息游戏。

即使 AI 明显更强，也不意味着：

AI 胜率 = 95%

很可能出现：

AI 强很多
但是：

Win  ≈ 65%
Draw ≈ 20%
Loss ≈ 15%

这是完全可能的。

原因是暗棋存在非常大的：

不可控随机性。

例如开局第一次翻子就可能导致完全不同的战略局面。

2015 年的实验甚至观察到，后手可能因为先手第一步必须翻棋而获得更多信息，从而形成先后手的不对称。
ResearchGate

因此：

单局胜负对实力的反映没有国际象棋那么直接。

评价暗棋 AI，Elo / 胜率 / 和棋率 / 先后手拆分必须一起看。

8. 这对你现在开发自己的 AI，其实是个非常好的消息

因为你真正想达到的目标不是：

“超过 CLAP_CDC。”

你最开始的实际目标其实更接近：

“让程序达到普通人的水平。”

那么从公开资料看，这个目标远比你昨天想象得容易。

甚至可以这样理解：

第一阶段
合理棋子价值
+
基本翻棋策略
+
Alpha-Beta / MCTS
+
循环处理

已经有机会达到：

普通玩家

第二阶段
Chance-aware MCTS
+
Progressive Bias
+
Quality-based Reward
+
Endgame Database

目标：

熟练玩家 / 强玩家

2015、2016 年的 DarkKnight / Diablo 工作已经证明这些方向有效。
科学直达
+1

第三阶段
AlphaZero
+
神经网络
+
MCTS
+
Self-play

目标：

Computer Olympiad 级别

这是 CLAP_CDC 证明过的路线。
Sage Journals

9. 所以你昨天的失败，其实不能说明 AlphaZero 路线不行

这一点我现在特别想纠正。

你昨天：

NN
↓
Self-play
↓
循环
↓
和棋
↓
几乎不进步

并不能推出：

“这种游戏不适合 AlphaZero。”

因为现实证据已经证明：

AlphaZero 可以在 Chinese Dark Chess 上成为冠军程序。 
Sage Journals

更合理的解释是：

AlphaZero路线 ✅
你的训练实现 ⚠️
训练参数 ⚠️
Search设计 ⚠️
Replay数据质量 ⚠️
终局/重复处理 ⚠️

也就是说，你的问题很可能不是方向错误，而是距离一套成熟的 Banqi AlphaZero 实现还有几个关键工程环节没有补齐。

10. 如果让我给现在的局面下一个判断

我会这样排：

对象	我对其水平的判断
随机 AI	极弱
简单启发式	普通人以下
合理 Alpha-Beta	普通人附近
DarkKnight / Diablo 式 NMCTS	明显强于普通人
历史 Computer Olympiad 顶级程序	非常强
2024 CLAP_CDC	当前公开比赛中的顶级程序
顶尖人类 vs CLAP_CDC	公开资料不足，无法可靠量化

所以回答你的原问题：

不像传统象棋那样，我们可以明确说“顶级 AI 已经远远超过顶尖职业棋手”。

目前更准确的是：

暗棋顶级 AI 已经强到足以长期统治计算机暗棋比赛，而且肯定超过普通玩家；但“顶级 AI 是否已经全面碾压顶尖人类高手”目前缺少公开、严格、足够大的直接对战数据。

而且这反而让你的项目变得很有意思——“普通人水平”是一个现实且可达的短期目标，“顶尖 AI”才是后面的研究目标。

对于你现在的开发，我会把**“普通人水平”定义成第一座里程碑**，而不是一开始就拿 CLAP_CDC 做目标。最科学的办法是建立一个 AI 评级梯：Random → Greedy → AlphaBeta → MCTS → 你的 NN → CLAP 风格 NN-MCTS，然后逐级通过对战测试。这样你下一次训练失败时，你会立刻知道是“没有超过 V1”，还是“搜索甚至还不如传统算法”，不会再出现不知道训练到底有没有方向的问题。

