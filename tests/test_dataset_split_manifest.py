"""切分清单落盘 / 冻结 test / 跨版本泄漏审计的回归测试。

背景（2026-09-17 复核）：同一批 .sav 被反复重导出成 p1_v1/v2/v3，
各版本用同一 seed 洗**不同长度**的列表 ⇒ 划分完全不同，旧版本 train 覆盖新版本 test
约 80%，而 metadata 只存 SHA-256、没有文件清单 ⇒ 重叠无法被审计发现
（详见 reviews/BC_ACCEPTANCE_VERDICT_2026-09-17.md）。

本测试固化三项修复：
1. `metadata.json` 必须落盘每个 split 的 .sav 文件名清单；
2. `frozen_test_files` 必须把冻结局整批排除出 train/val（leak_free=True）；
3. `_reconstruct_legacy_manifests` 只在**逐项吻合**时落盘（宁可没有清单，
   也不能写错清单 —— 错清单会让审计给出虚假的"无泄漏"结论）；
4. `scripts/audit_dataset_leakage.py` 能识别内部重叠、跨版本泄漏，
   且**不会**把"零个可审计数据集"包装成"无泄漏"。
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import random
import shutil
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from junqi.dataset import (_reconstruct_legacy_manifests, export_replay_dataset,
                           load_frozen_test_files, write_json_file)


def _load_audit_module():
    """按文件路径导入 scripts/audit_dataset_leakage.py（scripts 不是包）。"""
    path = os.path.join(BASE, "scripts", "audit_dataset_leakage.py")
    spec = importlib.util.spec_from_file_location("audit_dataset_leakage", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_entries(n: int, plies_per_game: int = 25):
    """伪造 valid_game_entries：只需要 basename 与 len(samples)。"""
    return [(f"g{i:03d}.sav", None, [0] * plies_per_game) for i in range(n)]


class TestLegacyManifestReconstruction(unittest.TestCase):
    """反推历史切分清单：必须"全等才落盘"。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "p1_vX")
        os.makedirs(self.dir, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _expected_split(self, n=30, plies=25, seed=2026, ratios=(0.8, 0.1, 0.1)):
        entries = _fake_entries(n, plies)
        rng = random.Random(seed)
        lst = list(entries)
        rng.shuffle(lst)
        n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
        return {"train": lst[:n_train],
                "val": lst[n_train:n_train + n_val],
                "test": lst[n_train + n_val:]}

    def test_writes_manifest_only_on_exact_match(self):
        sp = self._expected_split()
        meta = {"version": "3.0.0", "seed": 2026, "valid_games": 30,
                "train_games": len(sp["train"]), "val_games": len(sp["val"]),
                "test_games": len(sp["test"]),
                "train_plies": sum(len(g) for _, _, g in sp["train"]),
                "val_plies": sum(len(g) for _, _, g in sp["val"]),
                "test_plies": sum(len(g) for _, _, g in sp["test"])}
        write_json_file(os.path.join(self.dir, "metadata.json"), meta)

        results = _reconstruct_legacy_manifests(_fake_entries(30), 2026,
                                                (0.8, 0.1, 0.1), [self.dir])
        self.assertTrue(results[0]["written"], results[0]["reason"])
        sidecar = os.path.join(self.dir, "split_files.json")
        self.assertTrue(os.path.exists(sidecar))
        with open(sidecar, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertTrue(payload["verified"])
        self.assertEqual(payload["files"]["train"],
                         sorted(os.path.basename(p) for p, _, _ in sp["train"]))
        self.assertEqual(payload["files"]["test"],
                         sorted(os.path.basename(p) for p, _, _ in sp["test"]))

    def test_refuses_to_write_when_plies_mismatch(self):
        """plies 对不上（说明有效局口径或 max_games 变了）绝不能落盘。"""
        sp = self._expected_split()
        meta = {"version": "3.0.0", "seed": 2026, "valid_games": 30,
                "train_games": len(sp["train"]), "val_games": len(sp["val"]),
                "test_games": len(sp["test"]),
                "train_plies": sum(len(g) for _, _, g in sp["train"]) + 1,   # 故意错 1
                "val_plies": sum(len(g) for _, _, g in sp["val"]),
                "test_plies": sum(len(g) for _, _, g in sp["test"])}
        write_json_file(os.path.join(self.dir, "metadata.json"), meta)

        results = _reconstruct_legacy_manifests(_fake_entries(30), 2026,
                                                (0.8, 0.1, 0.1), [self.dir])
        self.assertFalse(results[0]["written"])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "split_files.json")))
        self.assertFalse(results[0]["checks"]["train"]["match"])

    def test_refuses_when_valid_games_mismatch(self):
        """有效局数不同 ⇒ 洗的是不同长度的列表 ⇒ 划分必然不同，禁止落盘。"""
        sp = self._expected_split()
        meta = {"version": "2.0.0", "seed": 2026, "valid_games": 835,   # p1_v2 的真实值
                "train_games": len(sp["train"]), "val_games": len(sp["val"]),
                "test_games": len(sp["test"]),
                "train_plies": sum(len(g) for _, _, g in sp["train"]),
                "val_plies": sum(len(g) for _, _, g in sp["val"]),
                "test_plies": sum(len(g) for _, _, g in sp["test"])}
        write_json_file(os.path.join(self.dir, "metadata.json"), meta)
        results = _reconstruct_legacy_manifests(_fake_entries(30), 2026,
                                                (0.8, 0.1, 0.1), [self.dir])
        self.assertFalse(results[0]["written"])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "split_files.json")))


class TestFrozenTestManifest(unittest.TestCase):
    """load_frozen_test_files 的读取语义。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_frozen_test_files(os.path.join(self.tmp.name, "no.json")), [])

    def test_reads_dict_and_bare_list(self):
        p1 = os.path.join(self.tmp.name, "a.json")
        write_json_file(p1, {"files": ["b.sav", "a.sav", "a.sav"],
                             "n_files": 2})
        self.assertEqual(load_frozen_test_files(p1), ["a.sav", "b.sav"])
        p2 = os.path.join(self.tmp.name, "b.json")
        write_json_file(p2, ["x.sav", "y.sav"])
        self.assertEqual(load_frozen_test_files(p2), ["x.sav", "y.sav"])

    def test_basename_normalization(self):
        p = os.path.join(self.tmp.name, "c.json")
        write_json_file(p, {"files": ["some/dir/z.sav"]})
        self.assertEqual(load_frozen_test_files(p), ["z.sav"])


class TestExportSplitManifestIntegration(unittest.TestCase):
    """真实 .sav 语料上的导出：清单落盘 + 冻结排除 + 内部互斥。

    只用 12 局小语料（复制到临时目录），避免拖慢测试套件。
    """

    @classmethod
    def setUpClass(cls):
        cls.sav_src = None
        for pat in ("军旗复盘/*.sav", "../军旗复盘/*.sav"):
            found = sorted(glob.glob(os.path.join(BASE, pat)))
            if found:
                cls.sav_src = found
                break

    def setUp(self):
        if not self.sav_src:
            self.skipTest("未找到 .sav 语料")
        self.tmp = tempfile.TemporaryDirectory()
        self.sav_dir = os.path.join(self.tmp.name, "savs")
        os.makedirs(self.sav_dir, exist_ok=True)
        for p in self.sav_src[:12]:
            shutil.copyfile(p, os.path.join(self.sav_dir, os.path.basename(p)))

    def tearDown(self):
        self.tmp.cleanup()

    def _export(self, out_name, **kw):
        out = os.path.join(self.tmp.name, out_name)
        stats = export_replay_dataset(self.sav_dir, out_dir=out, seed=2026,
                                      version="9.0.0", min_plies=20, **kw)
        return out, stats

    def test_manifest_written_and_internally_disjoint(self):
        out, stats = self._export("plain")
        meta_path = os.path.join(out, "metadata.json")
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertIn("split_files", meta)
        files = meta["split_files"]
        self.assertEqual(sorted(files.keys()), ["test", "train", "val"])
        # 内部互斥
        self.assertEqual(set(files["train"]) & set(files["val"]), set())
        self.assertEqual(set(files["train"]) & set(files["test"]), set())
        self.assertEqual(set(files["val"]) & set(files["test"]), set())
        # 清单必须覆盖全部有效局（否则审计会漏看）
        self.assertEqual(len(files["train"]) + len(files["val"]) + len(files["test"]),
                         stats.valid_games)
        self.assertFalse(meta["leak_free"])
        self.assertEqual(len(stats.split_files["train"]), stats.train_games)

    def test_frozen_test_excluded_from_train_val(self):
        out0, stats0 = self._export("plain2")
        with open(os.path.join(out0, "metadata.json"), "r", encoding="utf-8") as f:
            frozen = json.load(f)["split_files"]["test"]
        self.assertTrue(frozen, "小语料的 test 划分不应为空")

        out, stats = self._export("frozen", frozen_test_files=frozen)
        with open(os.path.join(out, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        files = meta["split_files"]
        frozen_set = set(frozen)
        self.assertTrue(meta["leak_free"])
        self.assertEqual(files["train"] and set(files["train"]) & frozen_set, set())
        self.assertEqual(set(files["val"]) & frozen_set, set())
        # 冻结模式下 test **恒等于** canonical 清单（含任何额外局都会重新引入跨版本重叠）
        self.assertEqual(set(files["test"]), frozen_set)
        self.assertEqual(meta["frozen_test_missing"], [])
        # 可训练池缩小 ⇒ train 局数不增加
        self.assertLessEqual(len(files["train"]), len(stats0.split_files["train"]))

    def test_no_write_npz_produces_nothing_on_disk(self):
        out, stats = self._export("nowrite", write_npz=False)
        self.assertFalse(os.path.exists(os.path.join(out, "metadata.json")))
        self.assertFalse(os.path.exists(os.path.join(out, "train.npz")))
        self.assertGreater(stats.valid_games, 0)


class TestLeakageAudit(unittest.TestCase):
    """审计脚本的判定逻辑。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_audit_module()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "datasets")
        os.makedirs(self.root, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_ds(self, name, version="3.0.0", leak_free=False,
                 train=(), val=(), test=()):
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        write_json_file(os.path.join(d, "metadata.json"), {
            "version": version, "seed": 2026, "min_plies": 20,
            "leak_free": leak_free,
            "train_games": len(train), "val_games": len(val), "test_games": len(test),
            "split_files": {"train": list(train), "val": list(val), "test": list(test)},
        })
        return d

    def _audit(self, dirs, canonical=None, exempt=None):
        datasets = [self.mod.load_dataset(d) for d in dirs]
        canonical = canonical or []
        if canonical:
            write_json_file(os.path.join(self.root, "canonical_test.json"),
                            {"files": canonical})
        return self.mod.audit(datasets, canonical, "datasets/canonical_test.json",
                              exempt or set())

    def test_clean_pair_passes(self):
        a = self._make_ds("A", train=["a1.sav"], val=["a2.sav"], test=["a3.sav"],
                          leak_free=True)
        b = self._make_ds("B", train=["b1.sav"], val=["b2.sav"], test=["a3.sav"],
                          leak_free=True)
        rep = self._audit([a, b], canonical=["a3.sav"])
        self.assertTrue(rep["passed"], rep["errors"])
        self.assertEqual(rep["n_auditable"], 2)

    def test_internal_overlap_is_error(self):
        a = self._make_ds("A", train=["x.sav"], val=["x.sav"], test=["y.sav"])
        rep = self._audit([a])
        self.assertFalse(rep["passed"])
        self.assertTrue(any("内部泄漏" in e for e in rep["errors"]))

    def test_cross_version_train_test_overlap_is_error(self):
        """核心回归：A 训过的局出现在 B 的 test ⇒ 必须判 ERROR。"""
        a = self._make_ds("A", train=["g1.sav"], val=[], test=["g9.sav"])
        b = self._make_ds("B", train=["g8.sav"], val=[], test=["g1.sav"])
        rep = self._audit([a, b])
        self.assertFalse(rep["passed"])
        self.assertTrue(any("重合" in e or "重叠" in e for e in rep["errors"]))
        pair = rep["cross"][0]
        self.assertEqual(pair["matrix"]["train&test"], 1)

    def test_cross_version_train_val_overlap_is_only_warning(self):
        """train/val 之间的重叠只影响模型选择可比性 ⇒ WARN 而非 ERROR。"""
        a = self._make_ds("A", train=["g1.sav"], val=["g2.sav"], test=["g9.sav"])
        b = self._make_ds("B", train=["g2.sav"], val=["g3.sav"], test=["g8.sav"])
        rep = self._audit([a, b])
        self.assertTrue(rep["passed"], rep["errors"])
        self.assertEqual(rep["cross"][0]["status"], "warn")
        self.assertTrue(any("模型选择不可比" in w for w in rep["warnings"]))

    def test_exempt_downgrades_to_warning(self):
        a = self._make_ds("p1_v3", version="3.0.0",
                          train=["g1.sav"], val=[], test=["g9.sav"])
        b = self._make_ds("p1_v4", version="4.0.0",
                          train=["g8.sav"], val=[], test=["g1.sav"])
        rep = self._audit([a, b], exempt={"p1_v3"})
        self.assertTrue(rep["passed"], rep["errors"])
        self.assertEqual(rep["cross"][0]["status"], "legacy-exempt")
        self.assertTrue(any("历史豁免" in w for w in rep["warnings"]))

    def test_zero_auditable_is_error_not_pass(self):
        """没有可审计对象 ≠ 无泄漏。"""
        a = os.path.join(self.root, "legacy")
        os.makedirs(a, exist_ok=True)
        write_json_file(os.path.join(a, "metadata.json"),
                        {"version": "1.0.0", "split_files": None})
        rep = self._audit([a])
        self.assertFalse(rep["passed"])
        self.assertEqual(rep["n_auditable"], 0)
        self.assertTrue(any("无法判定" in e for e in rep["errors"]))

    def test_canonical_in_train_is_error(self):
        a = self._make_ds("A", train=["c1.sav"], val=[], test=["z.sav"])
        rep = self._audit([a], canonical=["c1.sav"])
        self.assertFalse(rep["passed"])
        self.assertTrue(any("canonical" in e for e in rep["errors"]))

    def test_freeze_from_stamps_source_dataset(self):
        d = self._make_ds("SRC", train=["t1.sav"], val=["v1.sav"], test=["e1.sav"])
        out = os.path.join(self.root, "canonical_test.json")
        payload = self.mod.freeze_from(d, out, stamp=True)
        self.assertEqual(payload["files"], ["e1.sav"])
        with open(os.path.join(d, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertTrue(meta["leak_free"])
        self.assertEqual(meta["frozen_test_files"], ["e1.sav"])
        self.assertEqual(meta["frozen_test_missing"], [])
        self.assertIn("leak_free_source", meta)


if __name__ == "__main__":
    unittest.main()
