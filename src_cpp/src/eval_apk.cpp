#include "eval_apk.h"
#include "constants.h"
#include <algorithm>

namespace junqi {

double eval_apk_pure(const JunqiBoard& board, Color my_color) {
    if (board.is_terminal()) {
        int w = board.winner;
        if (w == -1 || w == -2) return 0.0;
        return (board.seat_color[w] == my_color) ? 500000.0 : -500000.0;
    }

    if (my_color == Color::NONE) {
        return 0.0;
    }

    Color opp_color = other_color(my_color);

    // 1. 动态炸弹定价：随存活敌方最高军衔缩放 (0x600ca 公式)
    double my_max_rank_val = 0.0;
    double opp_max_rank_val = 0.0;

    for (int pos = 0; pos < NUM_CELLS; ++pos) {
        const Piece& pc = board.cells[pos];
        if (pc.is_empty() || !pc.revealed) continue;

        double r_val = get_apk_piece_value(pc.rank);
        if (pc.color == my_color && r_val > my_max_rank_val) {
            my_max_rank_val = r_val;
        } else if (pc.color == opp_color && r_val > opp_max_rank_val) {
            opp_max_rank_val = r_val;
        }
    }

    double my_bomb_val = std::max(160.0, std::min(853.3333333333334, opp_max_rank_val * (1.0 / 3.0)));
    double opp_bomb_val = std::max(160.0, std::min(853.3333333333334, my_max_rank_val * (1.0 / 3.0)));

    double my_score = 0.0;
    double opp_score = 0.0;

    // 2. 遍历已知明子计算物质与行营占位加分
    for (uint8_t pos = 0; pos < NUM_CELLS; ++pos) {
        const Piece& pc = board.cells[pos];
        if (pc.is_empty() || !pc.revealed) continue;

        double val = (pc.rank == Rank::ZHA)
            ? ((pc.color == my_color) ? my_bomb_val : opp_bomb_val)
            : get_apk_piece_value(pc.rank);

        double pos_bonus = get_camp_position_bonus(pos);

        if (pc.color == my_color) {
            my_score += (val + pos_bonus);
        } else {
            opp_score += (val + pos_bonus);
        }
    }

    // 3. 地雷护旗防御阵地加成 (0x124094 +80)
    const auto& road_nb = get_road_neighbors();

    for (uint8_t hq_pos : HQS) {
        const Piece& flag_pc = board.cells[hq_pos];
        if (!flag_pc.is_empty() && flag_pc.revealed && flag_pc.rank == Rank::QI) {
            // 检查大本营周围是否有友方地雷
            bool has_mine_guard = false;
            const auto& nb = road_nb[hq_pos];
            for (int i = 0; i < nb.count; ++i) {
                const Piece& adj = board.cells[nb.neighbors[i]];
                if (!adj.is_empty() && adj.revealed && adj.color == flag_pc.color && adj.rank == Rank::LEI) {
                    has_mine_guard = true;
                    break;
                }
            }

            if (has_mine_guard) {
                if (flag_pc.color == my_color) {
                    my_score += MINE_FLAG_GUARD_BONUS;
                } else {
                    opp_score += MINE_FLAG_GUARD_BONUS;
                }
            }
        }
    }

    return my_score - opp_score;
}

double eval_apk_flip_root(const JunqiBoard& board, const Action& flip_act, Color my_color) {
    // 基础分继承当前全盘纯净估值
    double score = eval_apk_pure(board, my_color);

    // 统计己方已翻开明子并检查是否有未进营明子可一步进空营
    int revealed_friendly = 0;
    bool has_camp_entrance_opportunity = false;
    const auto& road_nb = get_road_neighbors();

    for (uint8_t i = 0; i < 60; ++i) {
        const Piece& p = board.cells[i];
        if (!p.is_empty() && p.revealed && p.color == my_color) {
            revealed_friendly++;
            if (!is_camp_idx(i)) {
                const auto& nb_i = road_nb[i];
                for (int j = 0; j < nb_i.count; ++j) {
                    uint8_t adj = nb_i.neighbors[j];
                    if (is_camp_idx(adj) && board.cells[adj].is_empty()) {
                        has_camp_entrance_opportunity = true;
                        break;
                    }
                }
            }
        }
    }

    // 1. 开局中心行营黄金暗子位强偏好 (0x5a3c0)
    // 严格约束：仅在开局全盘无己方明子、处纯开局盲翻探索期时赋予 +150
    if (revealed_friendly == 0) {
        for (uint8_t golden_pos : CENTER_CAMP_FLIPS) {
            if (flip_act.frm == golden_pos) {
                score += 150.0;
                break;
            }
        }
    }

    // 2. 首翻即据点，依托行营辐射拓荒 (0x5a3c0)
    bool has_friendly_camp = false;
    bool has_opp_threat = false;
    Color opp_color = (my_color == Color::RED) ? Color::BLUE : Color::RED;

    const auto& nb = road_nb[flip_act.frm];
    for (int i = 0; i < nb.count; ++i) {
        uint8_t adj = nb.neighbors[i];
        const Piece& cp = board.cells[adj];
        if (!cp.is_empty() && cp.revealed) {
            if (cp.color == my_color && is_camp_idx(adj)) {
                has_friendly_camp = true;
            } else if (cp.color == opp_color && !is_camp_idx(adj) && cp.rank != Rank::LEI && cp.rank != Rank::QI) {
                has_opp_threat = true;
            }
        }
    }

    if (has_friendly_camp) {
        score += 120.0;
    } else if (is_camp_idx(flip_act.frm)) {
        score += 60.0;
    }

    if (has_opp_threat && !has_friendly_camp) {
        score -= 40.0;
    }

    // 3. 战术纪律：若场上有未保护的己方明子且近邻有空营可进，严禁弃营盲目远端翻棋！
    if (has_camp_entrance_opportunity && !has_friendly_camp) {
        score -= 200.0;
    } else {
        score += 20.0;
    }

    return score;
}

} // namespace junqi
