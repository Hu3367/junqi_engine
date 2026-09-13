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

    lg = sub.add_parser("label_gui", help="启动实战交火打标与核验 GUI 工作台")
    lg.add_argument("--data", default="datasets/distill_tactical_labeled.json",
                    help="待核验打标数据集 JSON 路径")

    rg = sub.add_parser("replay_gui", help="启动对局复盘与决策点评工作台")
    rg.add_argument("--file", "-f", default=None, help="初始载入的复盘文件 (.sav 或 .json)")

    sr = sub.add_parser("summarize_reviews", help="一键全量汇总复盘点评并导出算法改进数据集")
    sr.add_argument("--storage", default="reviews/annotations.json", help="点评数据存储路径")
    sr.add_argument("--out-report", default="reports/review_summary_report.md", help="输出诊断报告路径")
    sr.add_argument("--out-dataset", default="datasets/user_review_labeled.json", help="输出算法改进数据集路径")
    sr.add_argument("--out-test", default="tests/test_user_reviewed_tactics.py", help="输出自动化回归测试代码路径")

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
    dv.add_argument("--data", default="datasets/distill_tactical_labeled.json",
                    help="预打标数据集 JSON 路径（若存在则优先加载）")
    dv.add_argument("--p1-dir", default=None,
                    help="P1: 使用 export_dataset 导出的 npz 数据集目录"
                         "（官方 list.cfg 真实终局标签，优先级高于 --data）")
    dv.add_argument("--samples", type=int, default=None, help="蒸馏局面数（默认使用全部数据或1200）")
    dv.add_argument("--epochs", type=int, default=8, help="训练轮数")
    dv.add_argument("--batch-size", type=int, default=128, help="批大小")
    dv.add_argument("--lr", type=float, default=5e-4, help="学习率")
    dv.add_argument("--val-ratio", type=float, default=0.15, help="验证集占比")
    dv.add_argument("--seed", type=int, default=2026, help="随机种子")
    dv.add_argument("--depth", type=int, default=3, help="专家搜索深度")
    dv.add_argument("--time-limit-ms", type=int, default=300, help="专家搜索单步时间预算（毫秒）")
    dv.add_argument("--device", default=None, help="计算设备")

    g = sub.add_parser("gate", help="P0: 统计严谨的模型晋级评测门控（配对同牌/座位互换/Wilson+SPRT）")
    g.add_argument("--a", default="hybrid2", help="候选策略 (random/greedy/search2/expert2/hybrid2/nn)")
    g.add_argument("--b", default="expert2", help="基准策略")
    g.add_argument("--seeds", type=int, default=100,
                   help="配对种子数（每种子先后手各一局；正式晋级建议 >=100 组即 200 局）")
    g.add_argument("--seed-base", type=int, default=2026, help="种子基数")
    g.add_argument("--workers", type=int, default=1, help="并行进程数")
    g.add_argument("--max-plies", type=int, default=None, help="和棋手数上限（默认引擎规则 1000）")
    g.add_argument("--model-a", default=None, help="候选策略模型权重路径")
    g.add_argument("--model-b", default=None, help="基准策略模型权重路径")
    g.add_argument("--elo0", type=float, default=0.0, help="SPRT H0 Elo 差")
    g.add_argument("--elo1", type=float, default=65.0, help="SPRT H1 Elo 差")
    g.add_argument("--out", default="reports", help="报告输出目录")

    ds = sub.add_parser("distill_search", help="P2: 搜索蒸馏（QSearch 教师软分布 -> Policy 头）")
    ds.add_argument("--base", default="models/bc_best.pt", help="基座模型路径")
    ds.add_argument("--out", default="models/search_distilled.pt", help="输出权重路径")
    ds.add_argument("--states", type=int, default=1200, help="蒸馏局面数")
    ds.add_argument("--eval-sets", nargs="*", default=None, help="附加固定评测集 jsonl")
    ds.add_argument("--epochs", type=int, default=6, help="训练轮数")
    ds.add_argument("--batch-size", type=int, default=128, help="批大小")
    ds.add_argument("--lr", type=float, default=3e-4, help="学习率")
    ds.add_argument("--depth", type=int, default=3, help="教师搜索深度")
    ds.add_argument("--time-limit-ms", type=int, default=300, help="教师单步时间预算（毫秒）")
    ds.add_argument("--temperature", type=float, default=120.0, help="教师软分布温度")
    ds.add_argument("--seed", type=int, default=2026, help="随机种子")
    ds.add_argument("--workers", type=int, default=0, help="教师打标并行进程数 (0=单进程)")
    ds.add_argument("--device", default=None, help="计算设备")

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
    default_sav = "军旗复盘" if os.path.exists("军旗复盘") else "../军旗复盘"
    ed.add_argument("--sav-dir", default=default_sav, help=".sav 复盘文件目录")
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
        from .gui import main as run_gui
        run_gui()
    elif args.cmd == "label_gui":
        from .label_gui import main as run_label_gui
        run_label_gui(["--data", args.data])
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
        if getattr(args, "p1_dir", None):
            from .train_value_distill import train_value_from_p1_dataset
            train_value_from_p1_dataset(p1_dir=args.p1_dir, base_model=args.base,
                                        out_path=args.out, epochs=args.epochs,
                                        batch_size=args.batch_size, lr=args.lr,
                                        seed=args.seed, device=args.device)
        else:
            from .train_value_distill import train_value_distill
            train_value_distill(base_model=args.base, out_path=args.out,
                                data_path=getattr(args, "data", None),
                                n_samples=args.samples, epochs=args.epochs,
                                batch_size=args.batch_size, lr=args.lr,
                                val_ratio=args.val_ratio, seed=args.seed,
                                depth=args.depth, time_limit_ms=args.time_limit_ms,
                                device=args.device)
    elif args.cmd == "gate":
        from .eval_gate import format_gate_report, run_gate
        seeds = list(range(args.seed_base, args.seed_base + args.seeds))
        rep = run_gate(args.a, args.b, seeds=seeds, workers=args.workers,
                       max_plies=args.max_plies, model_a=args.model_a,
                       model_b=args.model_b, elo0=args.elo0, elo1=args.elo1,
                       out_dir=args.out)
        print(format_gate_report(rep))
        print(f"JSON 已保存: {os.path.join(args.out, f'gate_{args.a}_vs_{args.b}.json')}")
    elif args.cmd == "distill_search":
        from .train_search_distill import train_search_distill
        train_search_distill(base_model=args.base, out_path=args.out,
                             n_states=args.states, eval_sets=args.eval_sets,
                             epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, depth=args.depth,
                             time_limit_ms=args.time_limit_ms,
                             temperature=args.temperature, seed=args.seed,
                             workers=args.workers, device=args.device)
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
    elif args.cmd == "replay_gui":
        from .replay_gui import run_replay_gui
        run_replay_gui(initial_file=args.file)
    elif args.cmd == "summarize_reviews":
        from .review_storage import ReviewStorage
        from .review_summary import ReviewSummaryPipeline
        storage = ReviewStorage(storage_path=args.storage)
        pipeline = ReviewSummaryPipeline(storage=storage)
        res = pipeline.run_all(
            out_report_path=args.out_report,
            out_dataset_path=args.out_dataset,
            out_test_path=args.out_test,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
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
