#include "expert_qsearch.h"

#include "rules.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace junqi {

namespace {

inline double piece_val(const ExpertWeights& w, Rank r) {
    unsigned i = static_cast<unsigned>(r);
    return (i < w.piece.size()) ? w.piece[i] : 0.0;
}

inline int rank_value(Rank r) { return static_cast<int>(r); }

ExpertState expert_state_from_board(const JunqiBoard& b) {
    ExpertState st;
    st.rank.fill(0);
    st.color.fill(0);
    st.revealed.fill(false);
    st.occupied.fill(false);
    for (int i = 0; i < NUM_CELLS; ++i) {
        const Piece& p = b.cells[i];
        if (p.is_empty()) continue;
        st.rank[i] = static_cast<uint8_t>(p.rank);
        st.color[i] = (p.color == Color::RED) ? 0 : 1;
        st.revealed[i] = p.revealed;
        st.occupied[i] = true;
        if (!p.revealed) {
            st.hidden_count++;
        } else {
            int k = rank_kind(st.rank[i]);
            if (k >= 0) st.revealed_count[st.color[i]][k]++;
        }
    }
    for (const Piece& p : b.dead) {
        int c = (p.color == Color::RED) ? 0 : 1;
        int k = rank_kind(static_cast<uint8_t>(p.rank));
        if (k >= 0) st.dead_count[c][k]++;
        st.total_dead++;
    }
    st.turn = b.turn;
    st.ply = b.ply;
    st.quiet = b.quiet;
    for (int s = 0; s < 2; ++s) {
        st.seat_color[s] = (b.seat_color[s] == Color::NONE) ? -1
                         : ((b.seat_color[s] == Color::RED) ? 0 : 1);
    }
    st.flag_needs_mines_cleared = b.cfg.flag_needs_mines_cleared;
    st.flag_gong_only = b.cfg.flag_gong_only;
    st.hq_locks_pieces = b.cfg.hq_locks_pieces;
    st.no_capture_draw_plies = b.cfg.no_capture_draw_plies;
    return st;
}

}  // namespace

// ---------------------------------------------------------------- 终局 / 走子

bool flag_attackable_no_hidden(const JunqiBoard& b) {
    // 对应 GameState._flag_attackable(hidden=[])
    if (b.cfg.flag_needs_mines_cleared) {
        Color enemy = other_color(b.my_color());
        int dead_mines = 0;
        for (const Piece& p : b.dead) {
            if (p.color == enemy && p.rank == Rank::LEI) dead_mines++;
        }
        if (dead_mines < 3) return false;
    }
    return true;
}

bool has_any_move(const JunqiBoard& b) {
    Color my = b.my_color();
    if (my == Color::NONE) return false;
    bool flag_ok = flag_attackable_no_hidden(b);

    for (int pos = 0; pos < NUM_CELLS; ++pos) {
        const Piece& pc = b.cells[pos];
        if (pc.is_empty() || !pc.revealed || pc.color != my) continue;
        if (pc.rank == Rank::LEI || pc.rank == Rank::QI) continue;
        if (b.cfg.hq_locks_pieces && is_hq_idx(pos)) continue;

        const CellNeighbors& nb = get_road_neighbors()[pos];
        for (int i = 0; i < nb.count; ++i) {
            uint8_t np = nb.neighbors[i];
            const Piece& t = b.cells[np];
            if (t.is_empty()) return true;
            if (b.is_attackable(pc.rank, t, np, flag_ok)) return true;
        }

        if (is_rail_idx(pos)) {
            // _rail_slides / _engineer_flights 在 attackable=lambda True 下：
            // 只要第一步是合法铁路格就必然产生至少一个落点。
            static const int dr[4] = {-1, 1, 0, 0};
            static const int dc[4] = {0, 0, 1, -1};
            int r = pos / COLS, c = pos % COLS;
            for (int d = 0; d < 4; ++d) {
                int nr = r + dr[d], nc = c + dc[d];
                if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) continue;
                uint8_t nidx = pos_to_idx(nr, nc);
                if (!is_rail_idx(nidx)) continue;
                if (is_cross_blocked(pos, nidx)) continue;
                return true;
            }
        }
    }
    return false;
}

void apply_expert(JunqiBoard& b, const Action& act) {
    int quiet = b.quiet + 1;
    b.ply += 1;
    int winner = b.winner;
    std::string reason = b.win_reason;
    const int mover_turn = static_cast<int>(b.turn);

    if (act.kind == ActionKind::FLIP) {
        b.cells[act.frm].revealed = true;
        if (!b.first_flip_done) {
            b.seat_color[b.turn] = b.cells[act.frm].color;
            b.seat_color[1 - b.turn] = other_color(b.cells[act.frm].color);
            b.first_flip_done = true;
        }
    } else {
        Piece mover = b.cells[act.frm];
        b.cells[act.frm] = Piece{};
        Piece target = b.cells[act.to];
        if (target.is_empty()) {
            b.cells[act.to] = mover;
        } else {
            quiet = 0;
            BattleResult res = resolve_battle(mover.rank, target.rank);
            if (res == BattleResult::ATTACKER_WINS) {
                b.dead.push_back(target);
                b.cells[act.to] = mover;
                if (target.rank == Rank::QI) {
                    winner = mover_turn;
                    reason = "flag";
                }
            } else if (res == BattleResult::BOTH_DIE) {
                b.cells[act.to] = Piece{};
                b.dead.push_back(target);
                b.dead.push_back(mover);
                // Python: 炸弹与旗同尽也判攻方胜（仅 flag_gong_only=False 时可达）
                if (target.rank == Rank::QI) {
                    winner = mover_turn;
                    reason = "flag";
                }
            } else {
                b.dead.push_back(mover);
            }
        }
    }

    b.quiet = quiet;
    b.winner = winner;
    b.win_reason = reason;
    b.turn = 1 - b.turn;

    if (b.winner == -2) {
        if (b.cfg.no_capture_draw_plies > 0 && b.quiet >= b.cfg.no_capture_draw_plies) {
            b.winner = -1;
            b.win_reason = "no_capture";
        } else if (b.ply >= b.cfg.max_plies) {
            b.winner = -1;
            b.win_reason = "max_plies";
        } else if (b.remaining_hidden_count() == 0 && !has_any_move(b)) {
            b.winner = mover_turn;
            b.win_reason = "immobilized";
        }
    }
}

// ---------------------------------------------------------------- 走法排序

double ExpertQSearch::_score_action(const JunqiBoard& b, const Action& act) const {
    Color my = b.my_color();

    if (act.kind == ActionKind::MOVE) {
        const Piece& target = b.cells[act.to];
        const Piece& mover_cell = b.cells[act.frm];
        Rank mover_rank = mover_cell.is_empty() ? Rank::PAI : mover_cell.rank;
        double attacker_val = piece_val(w, mover_rank);

        if (!target.is_empty() && target.revealed && target.color != my) {
            BattleResult res = resolve_battle(mover_rank, target.rank);
            double victim_val = piece_val(w, target.rank);

            double camp_outstrike_bonus = 0.0;
            bool camp_lose_risk = false;
            if (is_camp_idx(act.frm)) {
                Color opp = other_color(my);
                bool enemy_can_enter = false;
                if (opp != Color::NONE) {
                    const CellNeighbors& nb = get_road_neighbors()[act.frm];
                    for (int i = 0; i < nb.count; ++i) {
                        uint8_t np = nb.neighbors[i];
                        if (np == act.to) continue;
                        const Piece& e = b.cells[np];
                        if (!e.is_empty() && e.revealed && e.color == opp &&
                            e.rank != Rank::LEI && e.rank != Rank::QI) {
                            enemy_can_enter = true;
                            break;
                        }
                    }
                    if (!enemy_can_enter && is_rail_idx(act.frm)) {
                        int fr = act.frm / COLS, fc = act.frm % COLS;
                        for (int p = 0; p < NUM_CELLS; ++p) {
                            if (p == static_cast<int>(act.to)) continue;
                            const Piece& pc = b.cells[p];
                            if (pc.is_empty() || !pc.revealed || pc.color != opp) continue;
                            if (!is_rail_idx(p)) continue;
                            if (pc.rank == Rank::LEI || pc.rank == Rank::QI) continue;
                            if (p / COLS == fr || p % COLS == fc) {
                                enemy_can_enter = true;
                                break;
                            }
                        }
                    }
                }
                if (enemy_can_enter) camp_lose_risk = true;
                else camp_outstrike_bonus = w.camp_outstrike_bias;
            }

            double bomb_suicide_bonus = 0.0;
            if (target.rank == Rank::ZHA &&
                (mover_rank == Rank::LIAN || mover_rank == Rank::PAI ||
                 mover_rank == Rank::GONG || mover_rank == Rank::YING ||
                 mover_rank == Rank::TUAN)) {
                bomb_suicide_bonus = w.bomb_suicide_exchange;
            }

            if (target.rank == Rank::QI) return 900'000.0;

            if (res == BattleResult::ATTACKER_WINS) {
                double s = 500'000.0 + camp_outstrike_bonus + victim_val * 100.0 - attacker_val;
                if (camp_lose_risk) s -= 100'000.0;
                return s;
            } else if (res == BattleResult::BOTH_DIE) {
                double s = 300'000.0 + camp_outstrike_bonus + bomb_suicide_bonus +
                           victim_val * 100.0 - attacker_val;
                if (camp_lose_risk) s -= 100'000.0;
                return s;
            }
            return -100'000.0 + victim_val - attacker_val;
        }

        if (is_camp_idx(act.to)) {
            if (is_camp_idx(act.frm)) return -200'000.0;
            double camp_prio = target.is_empty() ? 250'000.0 : 150'000.0;
            Color opp = other_color(my);
            double pin_bomb_bonus = 0.0;
            if (opp != Color::NONE && target.is_empty()) {
                const CellNeighbors& nb = get_road_neighbors()[act.to];
                for (int i = 0; i < nb.count; ++i) {
                    const Piece& e = b.cells[nb.neighbors[i]];
                    if (!e.is_empty() && e.revealed && e.color == opp && e.rank == Rank::ZHA) {
                        pin_bomb_bonus = 50'000.0;
                        break;
                    }
                }
            }
            return camp_prio + pin_bomb_bonus + piece_val(w, mover_rank);
        }

        if (is_camp_idx(act.frm) && !is_camp_idx(act.to)) {
            if (mover_rank == Rank::ZHA) return -350'000.0;
            if (rank_value(mover_rank) >= rank_value(Rank::SHI)) return -250'000.0;
            Color opp = other_color(my);
            if (opp != Color::NONE) {
                const CellNeighbors& nb = get_road_neighbors()[act.frm];
                for (int i = 0; i < nb.count; ++i) {
                    const Piece& e = b.cells[nb.neighbors[i]];
                    if (e.is_empty() || !e.revealed || e.color != opp) continue;
                    if (e.rank == Rank::LEI || e.rank == Rank::QI) continue;
                    BattleResult r = resolve_battle(e.rank, mover_rank);
                    if (r == BattleResult::ATTACKER_WINS || r == BattleResult::BOTH_DIE) {
                        return -350'000.0;
                    }
                }
            }
            return -200'000.0;
        }

        if (!is_camp_idx(act.frm) && !is_camp_idx(act.to) &&
            rank_value(mover_rank) >= rank_value(Rank::SHI)) {
            bool reaches_empty_camp = false;
            const CellNeighbors& nb = get_road_neighbors()[act.to];
            for (int i = 0; i < nb.count && !reaches_empty_camp; ++i) {
                uint8_t n1 = nb.neighbors[i];
                if (is_camp_idx(n1) && b.cells[n1].is_empty()) {
                    reaches_empty_camp = true;
                    break;
                }
                if (b.cells[n1].is_empty()) {
                    const CellNeighbors& nb2 = get_road_neighbors()[n1];
                    for (int j = 0; j < nb2.count; ++j) {
                        uint8_t cp = nb2.neighbors[j];
                        if (is_camp_idx(cp) && b.cells[cp].is_empty()) {
                            reaches_empty_camp = true;
                            break;
                        }
                    }
                }
            }
            if (reaches_empty_camp) {
                Color opp = other_color(my);
                bool is_safe = true;
                if (opp != Color::NONE) {
                    for (int i = 0; i < nb.count; ++i) {
                        const Piece& e = b.cells[nb.neighbors[i]];
                        if (e.is_empty() || !e.revealed || e.color != opp) continue;
                        if (e.rank == Rank::LEI || e.rank == Rank::QI) continue;
                        BattleResult r = resolve_battle(e.rank, mover_rank);
                        if (r == BattleResult::ATTACKER_WINS || r == BattleResult::BOTH_DIE) {
                            is_safe = false;
                            break;
                        }
                    }
                }
                if (is_safe) return 200'000.0 + attacker_val * 100.0;
            }
        }

        // killer / history 不参与（见头文件说明）
        int r = act.to / COLS, c = act.to % COLS;
        double center_bias = 4.0 - std::abs(static_cast<double>(r) - 5.5) -
                             std::abs(static_cast<double>(c) - 2.0);
        return 1000.0 + center_bias * 10.0;
    }

    if (act.kind == ActionKind::FLIP) {
        int pos = act.frm;
        int r = pos / COLS, c = pos % COLS;
        Color opp = other_color(my);

        double camp_expansion_bonus = 0.0;
        bool has_friendly_camp = false;
        int friendly_camp_combat_rank = 0;
        bool has_safe_empty_camp = false;
        const CellNeighbors& nb = get_road_neighbors()[pos];
        for (int i = 0; i < nb.count; ++i) {
            uint8_t np = nb.neighbors[i];
            if (!is_camp_idx(np)) continue;
            const Piece& cb = b.cells[np];
            if (!cb.is_empty() && cb.revealed && cb.color == my) {
                has_friendly_camp = true;
                if (cb.rank != Rank::LEI && cb.rank != Rank::QI) {
                    friendly_camp_combat_rank =
                        std::max(friendly_camp_combat_rank, rank_value(cb.rank));
                }
            } else if (cb.is_empty()) {
                bool enemy_around_camp = false;
                const CellNeighbors& nb2 = get_road_neighbors()[np];
                for (int j = 0; j < nb2.count; ++j) {
                    const Piece& e = b.cells[nb2.neighbors[j]];
                    if (!e.is_empty() && e.revealed && e.color == opp) {
                        enemy_around_camp = true;
                        break;
                    }
                }
                if (!enemy_around_camp) has_safe_empty_camp = true;
            }
        }

        if (has_friendly_camp) {
            double rank_boost = (friendly_camp_combat_rank >= rank_value(Rank::SHI))
                                    ? 15'000.0 : 5'000.0;
            camp_expansion_bonus = 100'000.0 + rank_boost;
            if (has_safe_empty_camp) camp_expansion_bonus += 20'000.0;
        } else if (has_safe_empty_camp) {
            camp_expansion_bonus = 40'000.0;
        }

        double territory_bias = 0.0;
        if ((r >= 2 && r <= 4) || (r >= 7 && r <= 9)) territory_bias = 25'000.0;
        else if (r == 5 || r == 6) territory_bias = 20'000.0;
        else if (r == 1 || r == 10) territory_bias = 5'000.0;
        else if (r == 0 || r == 11) territory_bias = -20'000.0;

        int friendly_guards = 0, enemy_threats = 0;
        for (int i = 0; i < nb.count; ++i) {
            const Piece& nbpc = b.cells[nb.neighbors[i]];
            if (nbpc.is_empty() || !nbpc.revealed) continue;
            if (nbpc.color == my) {
                if (rank_value(nbpc.rank) >= rank_value(Rank::SHI)) friendly_guards += 2;
                else if (nbpc.rank != Rank::LEI && nbpc.rank != Rank::QI) friendly_guards += 1;
            } else if (nbpc.color == opp) {
                if (rank_value(nbpc.rank) >= rank_value(Rank::SHI)) enemy_threats += 2;
                else if (nbpc.rank != Rank::LEI && nbpc.rank != Rank::QI) enemy_threats += 1;
            }
        }

        if (territory_bias < -50'000.0 && friendly_guards > 0) {
            territory_bias = 10'000.0 * friendly_guards;
        }

        double safety_score;
        if (friendly_guards > enemy_threats) {
            safety_score = 60'000.0 + (friendly_guards - enemy_threats) * 5000.0;
        } else if (enemy_threats > friendly_guards) {
            safety_score = 500.0 - enemy_threats * 1000.0;
        } else {
            safety_score = 20'000.0;
        }
        return safety_score + camp_expansion_bonus + territory_bias;
    }

    return 0.0;
}

// ---------------------------------------------------------------- QSearch

double ExpertQSearch::_qsearch(JunqiBoard& b, double alpha, double beta, int depth_left) {
    stats.qnodes++;

    if (b.is_terminal()) {
        if (b.winner == -1) return 0.0;
        double win = EXPERT_WIN_SCORE - b.ply;
        return (b.winner == static_cast<int>(b.turn)) ? win : -win;
    }

    ExpertState est = expert_state_from_board(b);
    double stand_pat = eval_expert_cpp(est, static_cast<int>(b.turn), w, false);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;
    if (depth_left <= 0) return stand_pat;

    double max_piece_val = 0.0;
    for (double v : w.piece) max_piece_val = std::max(max_piece_val, v);
    double big_delta = max_piece_val + 200.0;
    if (stand_pat + big_delta < alpha) return alpha;

    Color my = b.my_color();
    std::vector<Action> acts = b.legal_actions();
    std::vector<Action> tactical;
    tactical.reserve(16);
    for (const Action& a : acts) {
        if (a.kind != ActionKind::MOVE) continue;
        const Piece& t = b.cells[a.to];
        if (!t.is_empty() && t.revealed && t.color != my) tactical.push_back(a);
    }
    if (tactical.empty()) return stand_pat;

    // Python 用 sorted(..., reverse=True)：稳定排序，并列保持生成序。
    std::stable_sort(tactical.begin(), tactical.end(),
                     [&](const Action& x, const Action& y) {
                         return _score_action(b, x) > _score_action(b, y);
                     });

    for (const Action& a : tactical) {
        const Piece& t = b.cells[a.to];
        if (!t.is_empty() && t.rank != Rank::QI) {
            double victim_val = piece_val(w, t.rank);
            if (stand_pat + victim_val + 50.0 < alpha) continue;
        }
        JunqiBoard child = b;
        apply_expert(child, a);
        double score = -_qsearch(child, -beta, -alpha, depth_left - 1);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

double ExpertQSearch::qsearch(const JunqiBoard& root, double alpha, double beta,
                              int depth_left) {
    JunqiBoard b = root;
    return _qsearch(b, alpha, beta, depth_left);
}

// ---------------------------------------------------------------- 紧凑字节入口

// blob 布局（与 junqi/core_bridge.py::encode_state_blob 严格对应）：
//   [0, 60)          : 每格一字节  rank | (color << 5) | (revealed ? 0x40 : 0)
//   [60, 60+nd)      : 每个阵亡子一字节  rank | (color << 5)
//   [60+nd, +18)     : 头部 "<7hi" —— turn, ply, quiet, seat0, seat1, flags,
//                      no_capture_draw_plies (int16) + max_plies (int32)
// flags 位：0 需挖完雷 1 需全翻 2 仅工兵扛旗 3 允许自杀攻击
//          4 大本营锁子 5 工兵铁路转弯 6 工兵可越子 7 首翻已完成
JunqiBoard board_from_blob(const std::string& blob) {
    const size_t n = blob.size();
    const size_t hdr = 18;
    size_t nd = (n >= 60 + hdr) ? (n - 60 - hdr) : 0;

    JunqiBoard b;
    b.cells.fill(Piece{});
    for (int i = 0; i < NUM_CELLS; ++i) {
        uint8_t code = static_cast<uint8_t>(blob[static_cast<size_t>(i)]);
        uint8_t rv = code & 0x1F;
        if (rv == 0) continue;
        b.cells[i] = Piece{static_cast<Rank>(rv),
                           ((code >> 5) & 1) ? Color::BLUE : Color::RED,
                           (code & 0x40) != 0};
    }
    for (size_t i = 0; i < nd; ++i) {
        uint8_t code = static_cast<uint8_t>(blob[60 + i]);
        uint8_t rv = code & 0x1F;
        if (rv == 0) continue;
        b.dead.push_back(Piece{static_cast<Rank>(rv),
                               ((code >> 5) & 1) ? Color::BLUE : Color::RED, true});
    }

    const char* hp = blob.data() + 60 + nd;
    auto rd16 = [&](int k) -> int16_t {
        int16_t v;
        std::memcpy(&v, hp + 2 * k, 2);
        return v;
    };
    b.turn = static_cast<uint8_t>(rd16(0));
    b.ply = rd16(1);
    b.quiet = rd16(2);
    int seat0 = rd16(3), seat1 = rd16(4);
    b.seat_color[0] = (seat0 < 0) ? Color::NONE : ((seat0 == 0) ? Color::RED : Color::BLUE);
    b.seat_color[1] = (seat1 < 0) ? Color::NONE : ((seat1 == 0) ? Color::RED : Color::BLUE);
    int flags = rd16(5);
    int ncp = rd16(6);
    int32_t max_plies;
    std::memcpy(&max_plies, hp + 14, 4);

    b.cfg.flag_needs_mines_cleared = (flags & 1) != 0;
    b.cfg.flag_needs_all_flipped = (flags & 2) != 0;
    b.cfg.flag_gong_only = (flags & 4) != 0;
    b.cfg.allow_suicide_attack = (flags & 8) != 0;
    b.cfg.hq_locks_pieces = (flags & 16) != 0;
    b.cfg.engineer_rail_turns = (flags & 32) != 0;
    b.cfg.engineer_can_fly_over_pieces = (flags & 64) != 0;
    b.first_flip_done = (flags & 128) != 0;
    b.cfg.no_capture_draw_plies = ncp;
    b.cfg.max_plies = max_plies;
    return b;
}

double ExpertQSearch::qsearch_blob(const std::string& blob, double alpha, double beta,
                                   int depth_left) {
    JunqiBoard b = board_from_blob(blob);
    return _qsearch(b, alpha, beta, depth_left);
}

}  // namespace junqi
