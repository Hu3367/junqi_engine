"""全局配置：规则开关、估值权重、搜索参数。

规则默认值的权威来源（2026-08-30 解包 base.apk 逆向确认，见
../apk_extracted/RULES_FROM_APK.md）：
  - App 内置帮助 data/junqi/ruleflip.txt（翻棋规则原文）
  - "规则设置"对话框 gamedlg.xml（和棋/大本营/吃地雷/吃军旗 选项）
  - .so 内文案（连续70步未吃子判和 / 总步数1000判和 / 相同局面循环判和）
"""
from dataclasses import dataclass


@dataclass
class RuleConfig:
    """翻棋细则开关（默认值按 APK 解包规则对齐，App 实测项已注明）。"""

    flag_needs_mines_cleared: bool = True   # APK：工兵挖完(3颗)地雷后才能扛军旗
    flag_needs_all_flipped: bool = False    # 场上所有棋子翻开后才能吃军旗（可选变体）
    flag_gong_only: bool = True             # APK：军旗只有工兵能吃（App"吃军旗:工兵"档）
    allow_suicide_attack: bool = False      # 允许小子撞大子（App 实测：不允许）
    hq_locks_pieces: bool = False           # 大本营锁死棋子（App 实测：翻棋不锁）
    engineer_rail_turns: bool = True        # 工兵铁路绕行（在铁路网内可任意转弯）
    engineer_can_fly_over_pieces: bool = False  # 工兵越子开关（默认 False：工兵不能越过轨道上的棋子移动）
    no_capture_draw_plies: int = 70         # APK：连续 70 步未吃子判和（0=关闭）
    max_plies: int = 1000                   # APK：双方总步数达到 1000 判和
    repetition_draw_count: int = 3          # APK：相同局面多次循环判和（对局层判定，
                                            #  按"可观察局面"出现次数计）


@dataclass
class SearchConfig:
    depth: int = 2          # 搜索深度（ ply ）
    samples: int = 6        # PIMC 采样的世界数 K
    time_limit_ms: int = 0  # >0 时迭代加深限时


@dataclass
class EvalWeights:
    piece: dict = None      # 棋子基础价值（Rank -> 分值）
    camp_occ: float = 10.0      # 占领行营（适度：防止 AI 只缩营不进攻）
    hq_locked: float = -8.0     # 非军旗子被困在大本营（仅 hq_locks_pieces 规则下有意义）
    flag_exposed: float = 40.0  # 己方军旗暴露在敌明子威胁下
    threat: float = 0.30        # 被敌明子威胁的子力折损系数
    attack: float = 0.25        # 威胁敌子的进攻机会系数
    attack_camp: float = 0.20   # 行营内发起的威胁（平衡威胁与真实吃子收益，杜绝缩营不杀）
    camp_siege: float = 3.0     # 营内子对邻格弱小敌子的围杀压力（行营围杀意识）
    # A2 增强项（基线 §4.2 第一条线：传统搜索显式判断）
    camp_zone: float = 2.0      # 行营势力：已方/敌方明子贴近空行营的净控制差（占营准备）
    fortress: float = 25.0      # 死区势能：双方 fortress_score 差（劣势方拖和潜力的估值注入）
    hidden_tempo: float = 6.0   # 暗子时差：活动明子数差（暗子激活需先翻后走两回合）
    # 2026-09-06 实证大数据增强项 (P1/P2: 梯队火力、占营胜率矩阵、小子拆弹与行营特权)
    echelon_si_compensation: float = 18.0   # 二线梯队火力网补偿（司令战死但拥有军长/双师/双炸时的抗悲观接管补偿）
    camp_matrix_weight: float = 12.0        # 占营比例非线性矩阵增益（反映 5:5 38% -> 8:2 83% 边际胜率阶跃）
    bomb_suicide_exchange: float = 150000.0 # 小子贴身拆弹在搜索排序中的战略特权加分 (47.5%炸中坚小子)
    camp_outstrike_bias: float = 400000.0   # 行营单向扑杀在走法排序中的特权加分 (开局50.1%吃子源自行营)
    camp_adjacent_flip_bias: float = 50000.0 # 据点邻域辐射翻棋启发加分 (96.2%邻营翻棋)

    def __post_init__(self):
        if self.piece is None:
            from .rules import Rank
            # 军旗的胜负价值由搜索的终局分体现，估值中只保留小额物质分，
            # 避免 1000 分进入暗子期望分摊后放大"翻子增值"假象。
            self.piece = {
                Rank.SI: 100, Rank.JUN: 90, Rank.SHI: 75, Rank.LV: 60,
                Rank.TUAN: 45, Rank.YING: 35, Rank.LIAN: 25, Rank.PAI: 18,
                Rank.GONG: 42, Rank.ZHA: 52, Rank.LEI: 30, Rank.QI: 50,
            }

    # ------------------------------------------------------- 序列化（tune 用）

    def to_dict(self) -> dict:
        return {"piece": {r.name: v for r, v in self.piece.items()},
                "camp_occ": self.camp_occ, "hq_locked": self.hq_locked,
                "flag_exposed": self.flag_exposed, "threat": self.threat,
                "attack": self.attack, "attack_camp": self.attack_camp,
                "camp_siege": self.camp_siege,
                "camp_zone": self.camp_zone, "fortress": self.fortress,
                "hidden_tempo": self.hidden_tempo,
                "echelon_si_compensation": self.echelon_si_compensation,
                "camp_matrix_weight": self.camp_matrix_weight,
                "bomb_suicide_exchange": self.bomb_suicide_exchange,
                "camp_outstrike_bias": self.camp_outstrike_bias,
                "camp_adjacent_flip_bias": self.camp_adjacent_flip_bias}

    @classmethod
    def from_dict(cls, d: dict) -> "EvalWeights":
        from .rules import Rank
        w = cls(camp_occ=d["camp_occ"], hq_locked=d["hq_locked"],
                flag_exposed=d["flag_exposed"], threat=d["threat"],
                attack=d["attack"], attack_camp=d["attack_camp"],
                camp_siege=d["camp_siege"])
        w.piece = {Rank[k]: v for k, v in d["piece"].items()}
        # A2 新键向后兼容：旧字典（无新键）加载时取默认值（类属性即 dataclass 默认）
        w.camp_zone = d.get("camp_zone", cls.camp_zone)
        w.fortress = d.get("fortress", cls.fortress)
        w.hidden_tempo = d.get("hidden_tempo", cls.hidden_tempo)
        w.echelon_si_compensation = d.get("echelon_si_compensation", cls.echelon_si_compensation)
        w.camp_matrix_weight = d.get("camp_matrix_weight", cls.camp_matrix_weight)
        w.bomb_suicide_exchange = d.get("bomb_suicide_exchange", cls.bomb_suicide_exchange)
        w.camp_outstrike_bias = d.get("camp_outstrike_bias", cls.camp_outstrike_bias)
        w.camp_adjacent_flip_bias = d.get("camp_adjacent_flip_bias", cls.camp_adjacent_flip_bias)
        return w
