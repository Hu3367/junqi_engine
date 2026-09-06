"""官方 APK 原生假想敌 (ApkNativeAgent) 命令行挑战评测脚本。

使用示例:
  # 使用专家搜索 Agent 对战 APK 中级假想敌 10 局
  venv\\Scripts\\python.exe scripts/challenge_apk.py --agent expert --games 10 --level intermediate

  # 使用混合引擎 HybridAgent 对战 APK 高级假想敌 40 局
  venv\\Scripts\\python.exe scripts/challenge_apk.py --agent hybrid --model models/best.pt --games 40 --level advanced
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from junqi.ai import Agent, ApkNativeAgent, ExpertAgent, HybridAgent, NNAgent
from junqi.benchmark import run_apk_challenge
from junqi.config import RuleConfig


def main():
    parser = argparse.ArgumentParser(description="自研 AI vs 官方 APK 原生假想敌挑战评测")
    parser.add_argument("--agent", choices=["expert", "hybrid", "search", "nn"], default="expert",
                        help="参评的自研智能体类型 (默认 expert，无需外部权重)")
    parser.add_argument("--level", choices=["beginner", "intermediate", "advanced"], default="intermediate",
                        help="官方 APK 假想敌难度 (默认 intermediate)")
    parser.add_argument("--games", type=int, default=10,
                        help="对抗对局数 (先后手各半，默认 10)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="基础发牌随机种子 (默认 2026)")
    parser.add_argument("--model", type=str, default="models/best.pt",
                        help="神经网络模型路径 (供 hybrid 或 nn 模式使用)")
    parser.add_argument("--depth", type=int, default=2,
                        help="自研专家搜索/混合引擎的战术搜索深度 (默认 2)")
    args = parser.parse_args()

    print(f"=== 正在初始化参评智能体: {args.agent.upper()} ===")
    if args.agent == "expert":
        candidate = ExpertAgent(seed=args.seed)
    elif args.agent == "hybrid":
        import os
        if not os.path.exists(args.model):
            print(f"[提示] 模型文件 '{args.model}' 不存在，自动回退到专家搜索引擎进行对抗。")
            candidate = ExpertAgent(seed=args.seed)
        else:
            candidate = HybridAgent(model_path=args.model, search_depth=args.depth, seed=args.seed)
    elif args.agent == "search":
        from junqi.config import SearchConfig
        candidate = Agent(search=SearchConfig(depth=args.depth), seed=args.seed)
    elif args.agent == "nn":
        candidate = NNAgent(model_path=args.model, simulations=50, seed=args.seed)
    else:
        candidate = ExpertAgent(seed=args.seed)

    print(f"=== 开始基准对抗: {args.agent} vs APK {args.level} (共 {args.games} 局) ===")
    t0 = time.time()
    report = run_apk_challenge(
        candidate_agent=candidate,
        n_games=args.games,
        level=args.level,
        base_seed=args.seed,
        cfg=RuleConfig()
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 55)
    print(f"           对战战报 (总耗时: {elapsed:.1f} 秒)")
    print("=" * 55)
    print(f" 对战双方     : {args.agent.upper()} vs 官方 APK 原生 AI ({args.level})")
    print(f" 总局数       : {report['games']} 局 (先后手各半)")
    print(f" 战绩         : {report['wins_a']} 胜 / {report['draws']} 和 / {report['wins_b']} 负")
    print(f" 得分率       : {report['score_rate_a']:.1%} (胜=1, 和=0.5, 负=0)")
    print(f" 和棋率       : {report['draw_rate']:.1%} (其中循环和棋: {report['repetition_draws']} 局)")
    print(f" 平均对局手数 : {report['avg_plies']:.1f} 手")
    print(f" 相对 Elo 分差: {report['elo_diff']:+.1f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
