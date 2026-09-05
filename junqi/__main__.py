"""命令行入口。

  python -m junqi calc                     局面计算器
  python -m junqi selfplay --games 200     自对弈批量研究
  python -m junqi test                     运行单元测试
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="junqi", description="军棋翻棋推演引擎")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("calc", help="交互式局面计算器")
    sub.add_parser("gui", help="人机对战图形界面")

    sp = sub.add_parser("selfplay", help="自对弈批量研究")
    sp.add_argument("--games", type=int, default=100, help="每组对局数")
    sp.add_argument("--a", default="greedy", help="先手策略: random/greedy/search2/search3")
    sp.add_argument("--b", default="random", help="后手策略")
    sp.add_argument("--workers", type=int, default=1, help="并行进程数")
    sp.add_argument("--seed", type=int, default=0, help="随机种子基数")
    sp.add_argument("--max-plies", type=int, default=None,
                    help="和棋手数上限（默认=引擎默认，APK 规则 1000）")
    sp.add_argument("--out", default="reports", help="报告输出目录")
    sp.add_argument("--no-swap", action="store_true", help="不跑先后手互换组")

    sp.add_argument("--eval-set", default=None, help="从 jsonl 评测集文件加载起始局面")
    sp.add_parser_model_a = sp.add_argument("--model-a", default=None, help="策略A模型权重文件路径")
    sp.add_parser_model_b = sp.add_argument("--model-b", default=None, help="策略B模型权重文件路径")

    tp = sub.add_parser("tune", help="估值权重调优（镜像自对弈爬山）")
    tp.add_argument("--rounds", type=int, default=2, help="爬山轮数")
    tp.add_argument("--candidates", type=int, default=6, help="每轮候选权重数")
    tp.add_argument("--games", type=int, default=40, help="每候选对局数（镜像）")
    tp.add_argument("--workers", type=int, default=1, help="并行进程数")
    tp.add_argument("--spec", default="greedy", help="对局策略: greedy/search2")
    tp.add_argument("--margin", type=float, default=0.06, help="显著改进阈值")
    tp.add_argument("--seed", type=int, default=0, help="随机种子")
    tp.add_argument("--max-plies", type=int, default=None,
                    help="和棋手数上限（默认=引擎默认 1000）")
    tp.add_argument("--out", default="reports/tune.json", help="结果输出 JSON")

    tr = sub.add_parser("train_rl", help="深度强化学习自对弈训练")
    tr.add_argument("--epochs", type=int, default=5, help="训练轮数")
    tr.add_argument("--games", type=int, default=40, help="每轮自对弈局数")
    tr.add_argument("--sims", type=int, default=60, help="MCTS 每手模拟次数")
    tr.add_argument("--eval-games", type=int, default=32,
                    help="门控评测：每阶段每方向局数（S1 起默认 32；正式晋级建议 ≥200）")
    tr.add_argument("--ref-games", type=int, default=6,
                    help="vs search2 参考对抗：每阶段局数（S1 起为晋升硬条件，0=关闭）")
    tr.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="并发 Worker 进程数")
    tr.add_argument("--curriculum-prob", type=float, default=0.3, help="残局课程采样概率")
    tr.add_argument("--midgame-prob", type=float, default=0.1,
                    help="S2：中盘评测集起始局面注入概率（开局多样性）")
    tr.add_argument("--batch-size", type=int, default=128, help="批处理大小")
    tr.add_argument("--lr", type=float, default=1e-3, help="学习率")
    tr.add_argument("--seed", type=int, default=42, help="随机种子")
    tr.add_argument("--out-dir", type=str, default="models", help="模型输出目录")
    tr.add_argument("--device", type=str, default=None, help="计算设备 (cuda/cpu)")
    tr.add_argument("--fresh", action="store_true", help="忽略已有检查点，从头训练")
    tr.add_argument("--rebase-baseline", action="store_true",
                    help="S0：用 bc_best.pt 重建发布基线（旧 best 备份），并清空候选进度")

    dv = sub.add_parser("distill_value", help="S2: 专家价值蒸馏预热（仅训练价值头）")
    dv.add_argument("--base", default="models/bc_best.pt", help="基座模型路径")
    dv.add_argument("--out", default="models/value_distilled.pt", help="输出权重路径")
    dv.add_argument("--samples", type=int, default=1200, help="蒸馏局面数")
    dv.add_argument("--epochs", type=int, default=3, help="训练轮数")
    dv.add_argument("--batch-size", type=int, default=128, help="批大小")
    dv.add_argument("--lr", type=float, default=5e-4, help="学习率")
    dv.add_argument("--val-ratio", type=float, default=0.15, help="验证集占比")
    dv.add_argument("--seed", type=int, default=2026, help="随机种子")
    dv.add_argument("--depth", type=int, default=3, help="专家搜索深度")
    dv.add_argument("--time-limit-ms", type=int, default=300, help="专家搜索单步时间预算（毫秒）")
    dv.add_argument("--device", default=None, help="计算设备")

    sub.add_parser("test", help="运行单元测试")

    an = sub.add_parser("analyze", help="分析人机对战记录（games/*.json）")
    an.add_argument("--dir", default="games")
    an.add_argument("--out", default="reports/human_analysis.md")

    rp = sub.add_parser("replay", help="解码并回放 App .sav 复盘文件")
    rp.add_argument("path", help=".sav 文件或包含 .sav 的目录")
    rp.add_argument("--no-check", action="store_true", help="跳过逐步合法性校验")
    rp.add_argument("--out", default="reports/replay_report.md", help="报告输出路径")

    fw = sub.add_parser("fit", help="从复盘数据行为克隆拟合估值权重")
    fw.add_argument("path", help=".sav 文件或包含 .sav 的目录")
    fw.add_argument("--points-per-game", type=int, default=20, help="每局采样决策点数")
    fw.add_argument("--max-cands", type=int, default=24, help="每决策点最多候选动作数")
    fw.add_argument("--epochs", type=int, default=60, help="训练轮数")
    fw.add_argument("--lr", type=float, default=0.5, help="学习率")
    fw.add_argument("--seed", type=int, default=0, help="随机种子")
    fw.add_argument("--out", default="reports/fit_weights.json", help="拟合权重输出")
    fw.add_argument("--verify", type=int, default=0, help="验证镜像对局数（0=跳过）")
    fw.add_argument("--workers", type=int, default=8, help="验证对局并行进程数")

    ed = sub.add_parser("export_dataset", help="P1: 从复盘数据导出标准行为克隆数据集")
    ed.add_argument("--sav-dir", default="../军旗复盘", help=".sav 复盘文件目录")
    ed.add_argument("--out-dir", default="datasets/p1_v1", help="输出 npz 目录")
    ed.add_argument("--seed", type=int, default=2026, help="随机种子")
    ed.add_argument("--version", default="1.0.0", help="数据集版本号")

    evd = sub.add_parser("eval_dataset", help="P1: 评估数据集上的 Policy 基准与分阶段覆盖率")
    evd.add_argument("--dataset", default="datasets/p1_v1/val.npz", help="npz 数据集文件路径")
    evd.add_argument("--samples", type=int, default=1000, help="评估样本量")

    tbc = sub.add_parser("train_bc", help="P2: 训练复盘行为克隆神经网络 (BC)")
    tbc.add_argument("--train-npz", default="datasets/p1_v1/train.npz", help="训练集 npz")
    tbc.add_argument("--val-npz", default="datasets/p1_v1/val.npz", help="验证集 npz")
    tbc.add_argument("--out", default="models/bc_best.pt", help="输出权重路径")
    tbc.add_argument("--epochs", type=int, default=15, help="训练轮数")
    tbc.add_argument("--batch-size", type=int, default=256, help="批大小")
    tbc.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    tbc.add_argument("--blocks", type=int, default=6, help="ResNet 块数")
    tbc.add_argument("--channels", type=int, default=128, help="隐藏通道数")
    tbc.add_argument("--device", default=None, help="设备")
    tbc.add_argument("--seed", type=int, default=2026, help="随机种子")

    bm = sub.add_parser("benchmark", help="P3.2: 运行固定残局靶场评测与多维指标看板输出")
    bm.add_argument("--model", default="models/bc_best.pt", help="待评测模型路径")
    bm.add_argument("--device", default="cpu", help="计算设备")
    bm.add_argument("--out-dir", default="metrics", help="看板输出目录")

    ebc = sub.add_parser("eval_bc", help="P2: 在独立测试集上评测 BC 模型")
    ebc.add_argument("--model", default="models/bc_best.pt", help="模型路径")
    ebc.add_argument("--test-npz", default="datasets/p1_v1/test.npz", help="测试集 npz")
    ebc.add_argument("--out", default="reports/p2_bc_report.md", help="输出 Markdown 报告路径")

    args = parser.parse_args(argv)

    if args.cmd == "calc":
        from .calculator import Calculator
        Calculator().loop()
    elif args.cmd == "gui":
        from .gui import main as gui_main
        gui_main()
    elif args.cmd == "selfplay":
        from .selfplay import run_selfplay
        result = run_selfplay(args.a, args.b, args.games, workers=args.workers,
                              base_seed=args.seed, out_dir=args.out,
                              max_plies=args.max_plies, also_swap=not args.no_swap,
                              model_path_a=args.model_a, model_path_b=args.model_b,
                              eval_set=args.eval_set)
        print(result["report_text"])
        print(f"\n明细 CSV: {result['csv']}\n报告: {result['report']}")
    elif args.cmd == "tune":
        from .tune import tune
        tune(rounds=args.rounds, candidates=args.candidates, games=args.games,
             workers=args.workers, spec=args.spec, margin=args.margin,
             seed=args.seed, out=args.out, max_plies=args.max_plies)
    elif args.cmd == "train_rl":
        from .train_rl import run_training
        run_training(epochs=args.epochs, games_per_epoch=args.games,
                     sims=args.sims, eval_games=args.eval_games,
                     ref_games=args.ref_games,
                     workers=args.workers, curriculum_prob=args.curriculum_prob,
                     midgame_prob=args.midgame_prob,
                     batch_size=args.batch_size, lr=args.lr,
                     seed=args.seed, out_dir=args.out_dir, device=args.device,
                     fresh=args.fresh, rebase_baseline=args.rebase_baseline)
    elif args.cmd == "distill_value":
        from .train_value_distill import train_value_distill
        train_value_distill(base_model=args.base, out_path=args.out,
                            n_samples=args.samples, epochs=args.epochs,
                            batch_size=args.batch_size, lr=args.lr,
                            val_ratio=args.val_ratio, seed=args.seed,
                            depth=args.depth, time_limit_ms=args.time_limit_ms,
                            device=args.device)
    elif args.cmd == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        ok = runner.run(suite).wasSuccessful()
        sys.exit(0 if ok else 1)
    elif args.cmd == "analyze":
        from .analyze import load_records, analyze
        records = load_records(args.dir)
        print(analyze(records, args.out))
        print(f"\n报告已保存: {args.out}")
    elif args.cmd == "replay":
        from .replay import load_dir, summarize
        games = load_dir(args.path, check=not args.no_check)
        report = summarize(games)
        print(report)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n报告已保存: {args.out}")
    elif args.cmd == "fit":
        from .fit_weights import run_fit
        run_fit(path=args.path, points_per_game=args.points_per_game,
                max_cands=args.max_cands, epochs=args.epochs, lr=args.lr,
                seed=args.seed, out=args.out, verify=args.verify,
                workers=args.workers)
    elif args.cmd == "export_dataset":
        from .dataset import export_replay_dataset
        stats = export_replay_dataset(args.sav_dir, out_dir=args.out_dir, seed=args.seed, version=args.version)
        print(f"P1 数据集已成功生成至 {args.out_dir}:")
        print(f"  - 总局数: {stats.total_sav_files} (有效: {stats.valid_games}, 无效: {stats.invalid_games})")
        print(f"  - 切分: Train {stats.train_games} 局 ({stats.train_plies} plies), Val {stats.val_games} 局 ({stats.val_plies} plies), Test {stats.test_games} 局 ({stats.test_plies} plies)")
        print(f"  - Policy 样本数: {stats.policy_samples_count}, Value 样本数: {stats.value_samples_count}")
        print(f"  - 阶段覆盖率: {stats.phase_distribution}")
    elif args.cmd == "eval_dataset":
        from .dataset import evaluate_dataset_policy
        report = evaluate_dataset_policy(args.dataset, max_samples=args.samples)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "train_bc":
        from .train_bc import train_bc
        train_bc(train_npz=args.train_npz, val_npz=args.val_npz, out_path=args.out,
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                 num_blocks=args.blocks, channels=args.channels, device=args.device,
                 seed=args.seed)
    elif args.cmd == "eval_bc":
        from .eval_bc import evaluate_test_set, generate_p2_report
        res = evaluate_test_set(model_path=args.model, test_npz=args.test_npz)
        generate_p2_report(res, out_md=args.out)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.cmd == "benchmark":
        from .benchmark import evaluate_net_benchmark, save_metrics_report
        from .net import JunqiNet
        if os.path.exists(args.model):
            net = JunqiNet.load_from_file(args.model, device=args.device)
            print(f"Loaded model from {args.model}")
        else:
            net = JunqiNet().to(args.device)
            print("Using initialized JunqiNet")
        metrics_res = evaluate_net_benchmark(net, device=args.device)
        print("\n=== Benchmark Evaluation Results ===")
        for k, v in metrics_res.items():
            print(f"  {k}: {v}")
        save_metrics_report({"value_mae": metrics_res}, output_dir=args.out_dir)
        print(f"\nDashboard saved to {args.out_dir}/benchmark_dashboard.md")


if __name__ == "__main__":
    main()
