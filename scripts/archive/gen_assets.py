"""生成 GUI 棋子贴图：从 APK 解包出的原版精灵表切片 + 缩放。

  python gen_assets.py

来源：base.apk assets/datax*.pak -> data/720/data/junqi/piece.png
（86x61/格，4列x7行：前3行红方 司令..军旗，中3行蓝方，末行 暗子背面/橙块/蓝块）。
输出 assets_app/cells/{r,b}_{RANK}.png + back.png，尺寸适配 gui.CELL（80x46）。
纯标准库实现（PNG 解码/编码 + 面积平均缩放），保持项目零第三方依赖。
"""
from __future__ import annotations

import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "assets_app", "piece_src.png")
OUT_DIR = os.path.join(HERE, "assets_app", "cells")

GRID_C, GRID_R = 4, 7          # 精灵表网格
TARGET_W, TARGET_H = 80, 46    # GUI 单元格内贴图尺寸

# 精灵表棋子顺序（行优先）：与 APK 内部棋子编码 1..12 一致
SPRITE_ORDER = ["SI", "JUN", "SHI", "LV", "TUAN", "YING", "LIAN", "PAI",
                "GONG", "ZHA", "LEI", "QI"]


def read_png(path):
    """解码 8-bit RGB (colortype 2) 非隔行 PNG -> (w, h, rows)。"""
    d = open(path, "rb").read()
    assert d[:8] == b"\x89PNG\r\n\x1a\n", "不是 PNG 文件"
    pos, idat = 8, b""
    w = h = None
    while pos < len(d):
        ln, tag = struct.unpack_from(">I4s", d, pos)
        data = d[pos + 8:pos + 8 + ln]
        if tag == b"IHDR":
            w, h, depth, ctype, _c, _f, inter = struct.unpack(">IIBBBBB", data)
            assert (depth, ctype, inter) == (8, 2, 0), \
                f"仅支持 8bit RGB 非隔行：depth={depth} ctype={ctype} inter={inter}"
        elif tag == b"IDAT":
            idat += data
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * 3
    rows, prev, p = [], bytearray(stride), 0
    for _ in range(h):
        f = raw[p]
        p += 1
        row = bytearray(raw[p:p + stride])
        p += stride
        if f == 1:
            for i in range(3, stride):
                row[i] = (row[i] + row[i - 3]) & 255
        elif f == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 255
        elif f == 3:
            for i in range(stride):
                a = row[i - 3] if i >= 3 else 0
                row[i] = (row[i] + ((a + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                a = row[i - 3] if i >= 3 else 0
                b = prev[i]
                c = prev[i - 3] if i >= 3 else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                row[i] = (row[i] + pred) & 255
        rows.append(row)
        prev = row
    return w, h, rows


def write_png(path, w, h, rows):
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def resize_box(rows, w, h, tw, th):
    """面积平均（box filter）缩放，缩小文字边缘比最近邻平滑。"""
    out = []
    for ty in range(th):
        y0, y1 = ty * h // th, max((ty + 1) * h // th, ty * h // th + 1)
        row = bytearray(tw * 3)
        for tx in range(tw):
            x0, x1 = tx * w // tw, max((tx + 1) * w // tw, tx * w // tw + 1)
            rs = gs = bs = n = 0
            for y in range(y0, y1):
                src = rows[y]
                for x in range(x0, x1):
                    rs += src[x * 3]
                    gs += src[x * 3 + 1]
                    bs += src[x * 3 + 2]
                    n += 1
            row[tx * 3:tx * 3 + 3] = bytes((rs // n, gs // n, bs // n))
        out.append(row)
    return out


def main():
    w, h, rows = read_png(SRC)
    cw, ch = w // GRID_C, h // GRID_R
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"源精灵表 {SRC}: {w}x{h}，格 {cw}x{ch} -> 缩放 {TARGET_W}x{TARGET_H}")

    def cell(gx, gy):
        x0, y0 = gx * cw, gy * ch
        sub = [rows[y][x0 * 3:(x0 + cw) * 3] for y in range(y0, y0 + ch)]
        return resize_box(sub, cw, ch, TARGET_W, TARGET_H)

    n = 0
    for i, rank in enumerate(SPRITE_ORDER):
        gy, gx = divmod(i, GRID_C)
        for color in ("r", "b"):
            write_png(os.path.join(OUT_DIR, f"{color}_{rank}.png"),
                      TARGET_W, TARGET_H, cell(gx, gy + (0 if color == "r" else 3)))
            n += 1
    write_png(os.path.join(OUT_DIR, "back.png"), TARGET_W, TARGET_H,
              cell(0, 6))                      # 末行第 1 格 = 暗子背面
    print(f"已生成 {n + 1} 张贴图 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
