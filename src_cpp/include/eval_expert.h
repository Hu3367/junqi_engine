#pragma once
// 专家级军棋翻棋评估函数的 C++ 移植（切片 1）。
//
// 对应 Python 真源：
//   junqi/eval_expert.py::evaluate_expert
//   junqi/analysis.py::fortress_score / is_dead_draw
//
// 逐位等价纪律：
//   1. board 遍历按 idx 升序（Python 按 dict 插入序）。实测两者差异
//      max_abs = 5.7e-14（default 标度）/ 2.3e-13（apk 标度），
//      远低于验收阈值 1e-9 —— 见 scratch/probe_eval_order_sensitivity.py。
//   2. alive counts 按 COMPOSITION 序（SI..QI，12 项）累加，与 Python dict 序一致。
//   3. NEIGHBORS / CAMPS 遍历序必须逐位一致（存在语义依赖，非仅浮点），
//      由 scripts/gen_expert_tables.py 从 Python 真源生成。
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "constants.h"
#include "types.h"

namespace junqi {

// COMPOSITION 序：SI JUN SHI LV TUAN YING LIAN PAI GONG ZHA LEI QI
constexpr int NUM_RANK_KINDS = 12;
constexpr uint8_t RANK_VALS[NUM_RANK_KINDS] = {13, 12, 11, 10, 9,  8,
                                               7,  6,  5,  20, 21, 22};
constexpr int RANK_COMP[NUM_RANK_KINDS] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 1};

// 索引常量，便于阅读
constexpr int R_SI = 0, R_JUN = 1, R_SHI = 2, R_LV = 3, R_TUAN = 4, R_YING = 5;
constexpr int R_LIAN = 6, R_PAI = 7, R_GONG = 8, R_ZHA = 9, R_LEI = 10, R_QI = 11;

inline int rank_kind(uint8_t rank_value) {
    switch (rank_value) {
        case 13: return R_SI;
        case 12: return R_JUN;
        case 11: return R_SHI;
        case 10: return R_LV;
        case 9:  return R_TUAN;
        case 8:  return R_YING;
        case 7:  return R_LIAN;
        case 6:  return R_PAI;
        case 5:  return R_GONG;
        case 20: return R_ZHA;
        case 21: return R_LEI;
        case 22: return R_QI;
        default: return -1;
    }
}

// ---------------------------------------------------------------- 权重

struct ExpertWeights {
    // 以 Rank 枚举值（0..22）为下标；未出现的值为 0。
    std::array<double, 32> piece{};

    double camp_occ{10.0};
    double hq_locked{-8.0};
    double flag_exposed{40.0};
    double threat{0.30};
    double attack{0.25};
    double attack_camp{0.20};
    double camp_siege{3.0};
    double camp_zone{2.0};
    double fortress{25.0};
    double hidden_tempo{6.0};
    double echelon_si_compensation{18.0};
    double camp_matrix_weight{12.0};
    double mine_flag_guard_bonus{80.0};
    double bomb_ratio{1.0 / 3.0};
    bool use_dynamic_bomb{true};

    // 走法排序专用（切片 2 的 _score_action 需要；与 EvalWeights 默认值一致）
    double bomb_suicide_exchange{150'000.0};
    double camp_outstrike_bias{400'000.0};
};

// ---------------------------------------------------------------- 局面

// Python GameState 的紧凑等价物（board/dead 构造后不可变）。
struct ExpertState {
    // 每格：rank 枚举值（0 = 空格）
    std::array<uint8_t, NUM_CELLS> rank{};
    // 每格颜色：0 = "r"(RED)，1 = "b"(BLUE)；空格无意义
    std::array<uint8_t, NUM_CELLS> color{};
    std::array<bool, NUM_CELLS> revealed{};
    std::array<bool, NUM_CELLS> occupied{};

    // 阵亡子按 (颜色, 兵种) 计数
    std::array<std::array<int, NUM_RANK_KINDS>, 2> dead_count{};
    int total_dead{0};

    // 明子按 (颜色, 兵种) 计数（含 hidden 的可见颜色？否，仅 revealed）
    std::array<std::array<int, NUM_RANK_KINDS>, 2> revealed_count{};

    int turn{0};
    int ply{0};
    int quiet{0};
    // -1 = 未定色，0 = "r"，1 = "b"
    std::array<int, 2> seat_color{-1, -1};

    bool flag_needs_mines_cleared{true};
    bool flag_gong_only{true};
    bool hq_locks_pieces{false};
    int no_capture_draw_plies{70};

    // 辅助
    int hidden_count{0};
};

// 由紧凑字节流构建 ExpertState。
// board_bytes: 60 字节，第 i 字节 = rank | (color << 5) | (revealed ? 0x40 : 0)
// dead_bytes : 每字节同上语法（无 revealed 位）
ExpertState make_expert_state(const std::string& board_bytes,
                              const std::string& dead_bytes,
                              int turn, int ply, int quiet,
                              int seat0, int seat1,
                              bool flag_needs_mines_cleared,
                              bool flag_gong_only, bool hq_locks_pieces,
                              int no_capture_draw_plies);

double fortress_score_cpp(const ExpertState& st, int seat);
bool is_dead_draw_cpp(const ExpertState& st, bool ignore_quiet_limit);

// 主入口：等价于 evaluate_expert(state, seat, w, ignore_rule_draw)
double eval_expert_cpp(const ExpertState& st, int seat,
                       const ExpertWeights& w, bool ignore_rule_draw);

// 供 Python 侧一致性测试读取的表（防止 C++ 表与 Python 真源漂移）
const std::array<CellNeighbors, NUM_CELLS>& get_expert_road_neighbors();
const std::array<uint8_t, 10>& get_expert_camp_order();

}  // namespace junqi
