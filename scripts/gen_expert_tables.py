"""生成 C++ 侧 evaluate_expert 所需的**顺序敏感**常量表。

为什么需要生成器（而不是手抄常量）：
`junqi/rules.py` 的 NEIGHBORS 由 `set` 迭代顺序决定，CAMPS 的遍历序由
`frozenset` 迭代顺序决定，两者都不是"自然顺序"。C++ 侧 `get_road_neighbors()`
按 (上,左,下,右) 构造，**与 Python 顺序不同**（例如 (0,1)：C++ 得 [0,6,2]，
Python 得 [0,2,6]）。而 `evaluate_expert` 里 `my_reach[0]` 直接取邻居列表首元素，
顺序差异会改变**语义**（不只是浮点误差）。故必须从 Python 真源生成。

用法:
    python scripts/gen_expert_tables.py [--check FILE]   # 默认写 src_cpp/src/eval_expert_tables.cpp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from junqi.rules import (CAMPS, COMPOSITION, NEIGHBORS, Rank,  # noqa: E402
                         ROWS, COLS)

HEADER = """// 本文件由 scripts/gen_expert_tables.py 自动生成，**请勿手工编辑**。
//
// 顺序敏感常量表：必须与 junqi/rules.py 的 NEIGHBORS / CAMPS 遍历序逐位一致。
// 生成理由见 scripts/gen_expert_tables.py 顶部注释。
#include "eval_expert.h"

namespace junqi {

"""

FOOTER = """
const std::array<CellNeighbors, NUM_CELLS>& get_expert_road_neighbors() {
    return EXPERT_ROAD_NEIGHBORS;
}

const std::array<uint8_t, 10>& get_expert_camp_order() {
    return EXPERT_CAMP_ORDER;
}

} // namespace junqi
"""


def gen() -> str:
    out = [HEADER]

    # ---- 公路邻接表（顺序 = Python NEIGHBORS 列表序）----
    out.append("// Python: junqi.rules.NEIGHBORS[idx] 的列表顺序（由 set 迭代序决定）\n")
    out.append("static constexpr std::array<CellNeighbors, NUM_CELLS> EXPERT_ROAD_NEIGHBORS = {{\n")
    for r in range(ROWS):
        for c in range(COLS):
            idx = r * COLS + c
            nbs = [n[0] * COLS + n[1] for n in NEIGHBORS[(r, c)]]
            assert len(nbs) <= 8, f"cell {(r, c)} has {len(nbs)} neighbors > 8"
            vals = ", ".join(str(n) for n in nbs)
            pad = ", ".join(["0"] * (8 - len(nbs)))
            body = f"{vals}, {pad}" if pad else vals
            out.append(f"    /* {idx:2d} (r{r},c{c}) */ {{{len(nbs)}, {{{body}}}}},\n")
    out.append("}};\n\n")

    # ---- 行营遍历序（= sorted(CAMPS, key=中营优先) 的稳定排序结果）----
    camps_sorted = sorted(CAMPS, key=lambda x: 0 if x in ((3, 2), (8, 2)) else 1)
    out.append("// Python: sorted(CAMPS, key=lambda c: 0 if c in ((3,2),(8,2)) else 1)\n")
    out.append("static constexpr std::array<uint8_t, 10> EXPERT_CAMP_ORDER = {\n    ")
    out.append(", ".join(str(r * COLS + c) for r, c in camps_sorted))
    out.append("\n};\n")

    out.append(FOOTER)
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    # 注意：这里必须用 store_true。用 nargs="?" + const=None 时，
    # 只写 --check 会令 args.check 恒为 None（`args.check is not None` 判不出），
    # 脚本会走"写入"分支而无法起到校验作用。
    ap.add_argument("--check", action="store_true",
                    help="只校验目标文件是否最新，不写入")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    text = gen()
    target = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "src_cpp" / "src" / "eval_expert_tables.cpp")

    if args.check:
        if not target.exists():
            print(f"MISSING {target}")
            sys.exit(1)
        if target.read_text(encoding="utf-8") != text:
            print(f"STALE {target} — 请运行 python scripts/gen_expert_tables.py")
            sys.exit(1)
        print(f"OK {target} up-to-date")
        return

    target.write_text(text, encoding="utf-8")
    print(f"wrote {target} ({len(text)} bytes)")
    # 附带打印 COMPOSITION 序，供 eval_expert.cpp 的 RANK_ORDER 对照
    print("COMPOSITION order:", [rk.name for rk in COMPOSITION])


if __name__ == "__main__":
    main()
