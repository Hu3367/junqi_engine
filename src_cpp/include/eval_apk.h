#pragma once

#include "board.h"

namespace junqi {

// 官方 0x124094 2 的幂次等比价值表
inline double get_apk_piece_value(Rank r) {
    switch (r) {
        case Rank::SI:   return 2560.0;
        case Rank::JUN:  return 1280.0;
        case Rank::SHI:  return 640.0;
        case Rank::LV:   return 320.0;
        case Rank::TUAN: return 160.0;
        case Rank::YING: return 80.0;
        case Rank::LIAN: return 40.0;
        case Rank::PAI:  return 30.0;
        case Rank::GONG: return 80.0;
        case Rank::ZHA:  return 426.0; // 动态基准
        case Rank::LEI:  return 70.0;
        case Rank::QI:   return 50.0;
        default:         return 0.0;
    }
}

// 60 格行营位置偏好加分 (0x1226c8)
inline double get_camp_position_bonus(uint8_t idx) {
    if (idx == pos_to_idx(3, 2) || idx == pos_to_idx(8, 2)) return 50.0; // 中营
    if (is_camp_idx(idx)) return 40.0;                                    // 角营
    return 0.0;
}

constexpr double MINE_FLAG_GUARD_BONUS = 80.0;

// 1:1 复刻 libjunqi.so 0x59f90 的极简纯净静态估值函数
double eval_apk_pure(const JunqiBoard& board, Color my_color);

// 根节点翻暗棋启发式独立打分 (0x5a3c0)
double eval_apk_flip_root(const JunqiBoard& board, const Action& flip_act, Color my_color);

} // namespace junqi
