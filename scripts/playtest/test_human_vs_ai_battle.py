"""
Junqi Human vs AI Battle with Strategy Analysis
人机对战 + 策略缺陷分析
"""

import json
from typing import Dict, List, Optional, Tuple


class BattleLogger:
    """战斗日志记录器"""
    
    def __init__(self):
        self.moves = []
        self.decisions = []
        self.strategic_issues = []
        
    def log_move(self, player_color: str, action: str, rationale: str):
        self.moves.append({
            "player": player_color,
            "action": action,
            "rationale": rationale
        })
        
    def log_decision(self, phase: str, selected_strategy: str, issue_type: str, 
                     description: str, severity: str = "medium"):
        self.decisions.append({
            "phase": phase,
            "strategy": selected_strategy,
            "issue": issue_type,
            "description": description,
            "severity": severity
        })
        
    def save_log(self, filename: str = "battle_analysis.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "moves": self.moves,
                "strategic_analysis": self.decisions,
                "document_gaps": self.find_document_gaps()
            }, f, ensure_ascii=False, indent=2)
            
    def find_document_gaps(self) -> List[Dict]:
        """自动识别文档章节缺失"""
        gaps = []
        
        for decision in self.decisions:
            if decision["severity"] == "high":
                gap = {
                    "chapter": self._map_phase_to_chapter(decision["phase"]),
                    "section": decision["strategy"],
                    "issue": decision["issue"],
                    "recommendation": self._generate_recommendation(decision)
                }
                gaps.append(gap)
                
        return gaps
        
    def _map_phase_to_chapter(self, phase: str) -> str:
        mapping = {
            "opening": "第 2 章",
            "midgame": "第 3 章", 
            "endgame": "第 4 章"
        }
        return mapping.get(phase, "未知")
        
    def _generate_recommendation(self, decision: Dict) -> str:
        """生成改进建议"""
        issues = decision["issue"]
        
        if "战术模式库" in issues:
            return "需要补充具体战术模式的算法实现代码"
        elif "动态价值评估" in issues:
            return "需要实现多维度局面评估函数"
        elif "路径规划" in issues:
            return "需要添加搜索算法支持特定战术"
        else:
            return f"针对{issues}问题补充可执行逻辑"


# ============ AI 对手实现（模拟我的决策过程）============

class MyAIPlayer:
    """代表"我"的 AI 玩家 - 会展示思考过程"""
    
    def __init__(self, color: str):
        self.color = color
        self.name = f"{'红方' if color=='r' else '黑方'}AI"
        self.thoughts = []
        
    def think(self, game_state: dict, opponent_move: Optional[str] = None) -> dict:
        """模拟 AI 的思考过程"""
        thought = {
            "step": len(self.thoughts) + 1,
            "current_position": game_state.get("turn"),
            "observation": [],
            "analysis": [],
            "decision_process": [],
            "final_choice": ""
        }
        
        # 观察当前局势
        hidden_count = game_state.get("hidden_count", 0)
        phase = self._detect_phase(hidden_count)
        
        thought["observation"].append(f"暗子数量：{hidden_count}")
        thought["observation"].append(f"游戏阶段：{phase}")
        
        # 基于阶段的分析
        if phase == "opening":
            analysis = self._opening_analysis(game_state)
            thought["analysis"].extend(analysis)
            thought["final_choice"] = self._opening_decide(game_state)
            
        elif phase == "midgame":
            analysis = self._midgame_analysis(game_state)
            thought["analysis"].extend(analysis)
            thought["final_choice"] = self._midgame_decide(game_state)
            
        else:  # endgame
            analysis = self._endgame_analysis(game_state)
            thought["analysis"].extend(analysis)
            thought["final_choice"] = self._endgame_decide(game_state)
            
        self.thoughts.append(thought)
        return thought
        
    def _detect_phase(self, hidden_count: int) -> str:
        if hidden_count >= 20:
            return "opening (开局)"
        elif hidden_count >= 6:
            return "midgame (中盘)"
        else:
            return "endgame (尾盘)"
            
    def _opening_analysis(self, game: dict) -> list:
        """开局阶段的思考（对应第 2 章）"""
        analyses = [
            "正在选择翻棋位置...",
            "考虑因素:",
            "- 安全性评分 (周围友军保护)",
            "- 战略价值 (前线优先)",
            "- 时差利用 (两回合优势)"
        ]
        return analyses
        
    def _midgame_analysis(self, game: dict) -> list:
        """中盘阶段的思考（对应第 3 章）"""
        analyses = [
            "正在进行子力交换决策...",
            "考虑因素:",
            "- MVV-LVA排序 (最值钱目标优先)",
            "- 炸弹使用时机",
            "- 行营争夺价值",
            "- 工兵挖雷优先级"
        ]
        return analyses
        
    def _endgame_analysis(self, game: dict) -> list:
        """尾盘阶段的思考（对应第 4 章）"""
        analyses = [
            "正在进行残局突破...",
            "考虑因素:",
            "- 死区完备度计算",
            "- 雷阵突破口选择",
            "- 逼和/拖和可能性",
            "- 封锁/反封锁战术"
        ]
        return analyses
        
    def _opening_decide(self, game: dict) -> str:
        return "选择翻前线安全位"
        
    def _midgame_decide(self, game: dict) -> str:
        return "执行吃子或关键占位"
        
    def _endgame_decide(self, game: dict) -> str:
        return "寻找突破口或构筑防守"


def interactive_game():
    """交互式人机对战 + 分析"""
    
    print("="*70)
    print("🎮 军棋翻棋 - 人机对战与策略分析")
    print("="*70)
    print("\n玩法说明:")
    print("1. 你会与 AI 进行对弈")
    print("2. AI 会展示每一步的思考过程")
    print("3. 对话结束后会生成策略分析报告")
    print("4. 报告将指出文档第 2-4 章的缺失和不足")
    print("\n操作指令:")
    print("- flip 翻子")
    print("- move from,to 移动棋子")
    print("- q 退出游戏")
    print("- ? 查看帮助")
    
    seed = input("\n请输入发牌种子（回车随机）: ").strip()
    seed = int(seed) if seed else 42
    
    print(f"\n使用种子：{seed}\n")
    print("请准备好你的棋盘...")
    input("按回车开始第一回合...")
    
    # 初始化
    battle_logger = BattleLogger()
    ai_player = MyAIPlayer('r')  # AI 执红先
    
    current_turn = 'r'  # 红先
    ply_num = 0
    game_over = False
    
    while not game_over:
        print(f"\n{'='*70}")
        print(f"【第{ply_num+1}回合】- {'红方' if current_turn=='r' else '黑方'}行动")
        print(f"{'='*70}")
        
        is_ai_turn = current_turn == 'r'
        
        if is_ai_turn:
            # AI 行动
            print(f"\n🤖 {ai_player.name} 思考中...")
            ai_thought = ai_player.think({"turn": current_turn})
            
            # 展示 AI 的思考
            print("\n💭 思考过程:")
            for obs in ai_thought["observation"]:
                print(f"  👁️ 观察到：{obs}")
                
            for analysis in ai_thought["analysis"]:
                print(f"  🧠 分析：{analysis}")
                
            print(f"\n✅ 最终决策：{ai_thought['final_choice']}")
            print(f"📋 AI 将执行该走法\n")
            
            # 记录 AI 的决策
            battle_logger.log_move(
                "AI", 
                ai_thought['final_choice'],
                f"[Phase: {ai_thought['observation'][1]}]"
            )
            
            # 切换回合
            current_turn = 'b'
            
        else:
            # 玩家行动
            print(f"\n👤 轮到你了!")
            print("\n输入命令:")
            print("  flip:(r,c)     - 翻开坐标(r,c)的暗子")
            print("  move:(r1,c1)->(r2,c2) - 移动棋子")
            print("  help           - 显示帮助")
            print("  quit           - 结束游戏")
            
            action = input("\n你的行动 > ").strip()
            
            if action.lower() == 'quit':
                print("游戏已退出")
                break
                
            elif action.lower() == 'help':
                continue
                
            elif action.startswith('flip'):
                pos_str = action.split(':')[1].strip()
                try:
                    coords = eval(pos_str)
                    print(f"你选择了翻开位置：{coords}")
                    
                    battle_logger.log_move(
                        "Human", 
                        f"flip:{coords}",
                        "自主选择翻棋位置"
                    )
                    
                    # 检测策略问题
                    strategy_issue = detect_strategic_issue(
                        coords, "opening", "翻棋决策"
                    )
                    if strategy_issue:
                        battle_logger.log_decision(
                            phase="opening",
                            selected_strategy="翻棋策略",
                            issue=strategy_issue["issue"],
                            description=strategy_issue["desc"],
                            severity="high"
                        )
                    
                except Exception as e:
                    print(f"❌ 解析失败：{e}")
                    continue
                    
            elif action.startswith('move'):
                parts = action.split('->')
                if len(parts) == 2:
                    try:
                        from_pos = eval(parts[0].split(':')[1])
                        to_pos = eval(parts[1])
                        print(f"你计划移动：{from_pos} → {to_pos}")
                        
                        battle_logger.log_move(
                            "Human", 
                            f"move:{from_pos}->{to_pos}",
                            "自主选择移动路径"
                        )
                        
                        # 检测策略问题
                        strategy_issue = detect_strategic_issue(
                            to_pos, "midgame", "移动决策"
                        )
                        if strategy_issue:
                            battle_logger.log_decision(
                                phase="midgame",
                                selected_strategy="移动策略",
                                issue=strategy_issue["issue"],
                                description=strategy_issue["desc"],
                                severity="medium"
                            )
                            
                    except Exception as e:
                        print(f"❌ 解析失败：{e}")
                        continue
            else:
                print("❌ 无效命令，请参考上方说明")
                continue
                
            # 切换回合
            current_turn = 'r'
            
        ply_num += 1
        
        if ply_num > 50:
            print("\n⚠️ 已达到最大回合数，进入分析阶段")
            break
            
    # 生成分析报告
    print("\n" + "="*70)
    print("📊 生成策略分析报告...")
    battle_logger.save_log("strategy_analysis_result.json")
    
    print("\n✅ 分析完成！报告已保存到 strategy_analysis_result.json")
    print("\n📝 核心发现摘要:")
    
    gaps = battle_logger.find_document_gaps()
    if gaps:
        print(f"\n发现了 {len(gaps)} 个文档缺陷:")
        for i, gap in enumerate(gaps, 1):
            print(f"\n{i}. 【{gap['chapter']}{gap['section']}】")
            print(f"   ❌ 问题：{gap['issue']}")
            print(f"   💡 建议：{gap['recommendation']}")
    else:
        print("\n目前未发现明显缺陷（需要更多对局验证）")
        
    print("\n请查看 strategy_analysis_result.json 获取详细分析")


def detect_strategic_issue(coords: tuple, phase: str, context: str) -> Optional[dict]:
    """检测策略是否合理（简单示例）"""
    # 这里可以添加更复杂的检测逻辑
    # 例如：检查翻棋位置是否在安全区域等
    
    r, c = coords
    
    # 简单规则：检查是否在铁路线上
    rail_rows = {1, 5, 6, 10}
    rail_cols = {0, 4}
    
    on_rail = r in rail_rows or c in rail_cols
    
    if context == "翻棋决策" and not on_rail:
        return {
            "issue": "翻棋策略缺乏铁路控制意识",
            "desc": f"位置({r},{c})不在铁路线上，应优先考虑铁路控制权"
        }
    
    return None


if __name__ == "__main__":
    interactive_game()
