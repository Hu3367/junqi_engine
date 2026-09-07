#pragma once

#include <array>
#include <vector>
#include "types.h"

namespace junqi {

// 10 个行营索引
constexpr std::array<uint8_t, 10> CAMPS = {
    11, 13, 17, 21, 23, 36, 38, 42, 46, 48
};

// 4 个大本营索引
constexpr std::array<uint8_t, 4> HQS = {
    1, 3, 56, 58
};

// 6 个开局中心行营黄金辐射位
constexpr std::array<uint8_t, 6> CENTER_CAMP_FLIPS = {
    16, 18, 22, 37, 41, 43
};

inline bool is_camp_idx(uint8_t idx) {
    for (uint8_t c : CAMPS) {
        if (c == idx) return true;
    }
    return false;
}

inline bool is_hq_idx(uint8_t idx) {
    for (uint8_t h : HQS) {
        if (h == idx) return true;
    }
    return false;
}

inline bool is_rail_idx(uint8_t idx) {
    int r = idx / COLS;
    int c = idx % COLS;
    if (r == 1 || r == 5 || r == 6 || r == 10) return true;
    if ((c == 0 || c == 4) && (r >= 1 && r <= 10)) return true;
    return false;
}

// 检查前线河流阻断 (5,1)-(6,1) 和 (5,3)-(6,3)
inline bool is_cross_blocked(uint8_t a, uint8_t b) {
    uint8_t p1 = pos_to_idx(5, 1);
    uint8_t p2 = pos_to_idx(6, 1);
    if ((a == p1 && b == p2) || (a == p2 && b == p1)) return true;

    uint8_t q1 = pos_to_idx(5, 3);
    uint8_t q2 = pos_to_idx(6, 3);
    if ((a == q1 && b == q2) || (a == q2 && b == q1)) return true;

    return false;
}

// 邻接表结构：最多 8 个邻居
struct CellNeighbors {
    uint8_t count{0};
    std::array<uint8_t, 8> neighbors{};
};

// 获取预计算的公路邻接表
const std::array<CellNeighbors, NUM_CELLS>& get_road_neighbors();

// 获取预计算的铁路正交邻接表
const std::array<CellNeighbors, NUM_CELLS>& get_rail_neighbors();

} // namespace junqi
