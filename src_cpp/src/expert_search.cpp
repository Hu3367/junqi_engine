#include "expert_search.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace junqi {

namespace {
constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

// 剩余暗子池：COMPOSITION − 明子 − 阵亡（按 Python Counter 的插入序：
// 先 "r" 后 "b"，每个颜色内按 COMPOSITION 序 SI..QI）
void remaining_types(const JunqiBoard& b, int rem[2][12], int* total) {
    int rev[2][12] = {};
    int dead[2][12] = {};
    for (int i = 0; i < NUM_CELLS; ++i) {
        const Piece& p = b.cells[i];
        if (p.is_empty() || !p.revealed) continue;
        int k = rank_kind(static_cast<uint8_t>(p.rank));
        if (k >= 0) rev[(p.color == Color::RED) ? 0 : 1][k]++;
    }
    for (const Piece& p : b.dead) {
        int k = rank_kind(static_cast<uint8_t>(p.rank));
        if (k >= 0) dead[(p.color == Color::RED) ? 0 : 1][k]++;
    }
    int t = 0;
    for (int c = 0; c < 2; ++c) {
        for (int k = 0; k < 12; ++k) {
            int v = RANK_COMP[k] - rev[c][k] - dead[c][k];
            rem[c][k] = v > 0 ? v : 0;
            t += rem[c][k];
        }
    }
    *total = t;
}

}  // namespace

// ---------------------------------------------------------------- 基础设施

void ExpertSearch::clear_heuristics() {
    killers_.fill({});
    killer_n_.fill(0);
    history_.fill(0);
}

void ExpertSearch::clear_tt() {
    if (tt_.empty()) {
        tt_.resize(1u << 18);
        tt_mask_ = tt_.size() - 1;
    }
    for (auto& e : tt_) e.used = false;
}

void ExpertSearch::set_deadline_ms(double ms) {
    if (ms > 0.0) {
        has_deadline_ = true;
        deadline_ = std::chrono::steady_clock::now().time_since_epoch() +
                    std::chrono::duration<double>(ms / 1000.0);
    } else {
        has_deadline_ = false;
    }
}

bool ExpertSearch::in_path(uint64_t key) const {
    for (uint64_t k : path_) {
        if (k == key) return true;
    }
    return false;
}

void ExpertSearch::tt_store(uint64_t key, int depth, double score, int flag,
                            const Action& mv) {
    if (tt_.empty()) clear_tt();
    TTEntry& e = tt_[key & tt_mask_];
    if (!e.used || e.key != key || depth >= e.depth) {
        e.key = key;
        e.depth = depth;
        e.score = score;
        e.flag = flag;
        e.used = true;
        e.has_move = true;
        e.move = mv;
    } else if (!e.has_move) {
        e.has_move = true;
        e.move = mv;
    }
}

bool ExpertSearch::tt_lookup(uint64_t key, int depth, double alpha, double beta,
                             bool* has_move, Action* mv, double* out_cut) {
    if (tt_.empty()) clear_tt();
    const TTEntry& e = tt_[key & tt_mask_];
    if (!e.used || e.key != key) return false;
    if (e.has_move) {
        *has_move = true;
        *mv = e.move;
    }
    if (e.depth >= depth) {
        stats.tt_hits++;
        if (e.flag == TT_FLAG_EXACT ||
            (e.flag == TT_FLAG_LOWER_BOUND && e.score >= beta) ||
            (e.flag == TT_FLAG_UPPER_BOUND && e.score <= alpha)) {
            *out_cut = e.score;
            return true;
        }
    }
    return false;
}

void ExpertSearch::update_cutoff_heuristics(const Action& a, int ply_depth, int depth) {
    if (ply_depth >= MAX_SEARCH_DEPTH) return;
    bool exists = false;
    for (int i = 0; i < killer_n_[ply_depth]; ++i) {
        if (killers_[ply_depth][i].kind == a.kind && killers_[ply_depth][i].frm == a.frm &&
            (a.kind != ActionKind::MOVE || killers_[ply_depth][i].to == a.to)) {
            exists = true;
            break;
        }
    }
    if (!exists) {
        // insert(0) + pop 到 MAX_KILLERS
        for (int i = MAX_KILLERS - 1; i > 0; --i) killers_[ply_depth][i] = killers_[ply_depth][i - 1];
        killers_[ply_depth][0] = a;
        if (killer_n_[ply_depth] < MAX_KILLERS) killer_n_[ply_depth]++;
    }
    size_t hk = static_cast<size_t>(a.frm) * NUM_CELLS + a.to;
    int v = history_[hk] + depth * depth;
    history_[hk] = v > MAX_HISTORY ? MAX_HISTORY : v;
}

double ExpertSearch::_killer_history_bonus(const Action& act) const {
    if (cur_ply_depth_ < MAX_SEARCH_DEPTH) {
        for (int i = 0; i < killer_n_[cur_ply_depth_]; ++i) {
            const Action& k = killers_[cur_ply_depth_][i];
            if (k.kind == act.kind && k.frm == act.frm &&
                (act.kind != ActionKind::MOVE || k.to == act.to)) {
                return 50'000.0;
            }
        }
    }
    if (act.kind == ActionKind::MOVE) {
        int h = history_[static_cast<size_t>(act.frm) * NUM_CELLS + act.to];
        if (h > 0) return 10'000.0 + static_cast<double>(h < MAX_HISTORY ? h : MAX_HISTORY);
    }
    return 0.0;
}

// ---------------------------------------------------------------- 走法排序

std::vector<Action> ExpertSearch::order_actions(const JunqiBoard& b,
                                                std::vector<Action>& acts,
                                                const OrderCtx& ctx) {
    if (acts.size() <= 1) return acts;
    cur_has_tt_ = ctx.has_tt_move;
    cur_tt_ = ctx.tt_move;
    cur_ply_depth_ = ctx.ply_depth;

    std::vector<Action> moves, flips;
    moves.reserve(acts.size());
    flips.reserve(acts.size());
    for (const Action& a : acts) {
        if (a.kind == ActionKind::MOVE) moves.push_back(a);
        else flips.push_back(a);
    }

    if (flips.size() > 5) {
        std::stable_sort(flips.begin(), flips.end(),
                         [&](const Action& x, const Action& y) {
                             return _score_action(b, x) > _score_action(b, y);
                         });
        size_t mf = moves.empty() ? 8 : 3;
        if (flips.size() > mf) flips.resize(mf);
    }

    std::vector<Action> merged = moves;
    merged.insert(merged.end(), flips.begin(), flips.end());
    std::stable_sort(merged.begin(), merged.end(),
                     [&](const Action& x, const Action& y) {
                         return _score_action(b, x) > _score_action(b, y);
                     });
    cur_has_tt_ = false;
    return merged;
}

// ---------------------------------------------------------------- 机会节点

double ExpertSearch::chance_flip_(JunqiBoard& b, uint8_t pos, int depth, int ply_depth,
                                  double alpha, double beta) {
    stats.chance_nodes++;

    int rem[2][12];
    int total_hidden = 0;
    remaining_types(b, rem, &total_hidden);

    if (total_hidden <= 0) {
        JunqiBoard child = b;
        apply_expert(child, Action::make_flip(pos));
        return -negamax_(child, depth - 1, ply_depth + 1, -beta, -alpha);
    }

    // 构造"翻开为 (color, rank)"的子局面（与 Python 完全一致：
    // **不走 apply**，只做首翻定色 + ply/quiet+1 + 困毙检查）
    auto make_child = [&](int c, int k) {
        JunqiBoard child = b;
        Color clr = (c == 0) ? Color::RED : Color::BLUE;
        child.cells[pos] = Piece{static_cast<Rank>(RANK_VALS[k]), clr, true};
        if (!child.first_flip_done) {
            child.seat_color[child.turn] = clr;
            child.seat_color[1 - child.turn] = other_color(clr);
            child.first_flip_done = true;
        }
        child.turn = 1 - child.turn;
        child.ply = b.ply + 1;
        child.quiet = b.quiet + 1;
        if (child.remaining_hidden_count() == 0 && !has_any_move(child)) {
            child.winner = static_cast<int>(b.turn);
            child.win_reason = "immobilized";
        }
        return child;
    };

    auto child_value = [&](const JunqiBoard& child) -> double {
        if (child.is_terminal()) {
            if (child.winner == -1) return 0.0;
            double win = EXPERT_WIN_SCORE - child.ply;
            return (child.winner == static_cast<int>(child.turn)) ? win : -win;
        }
        return eval_expert_board(child, static_cast<int>(child.turn), w, false);
    };

    // 几率前沿截断：深层直接求解析期望，杜绝 (24)^d 组合爆炸
    if (depth <= 1 || ply_depth >= 1) {
        double expected = 0.0;
        for (int c = 0; c < 2; ++c) {
            for (int k = 0; k < 12; ++k) {
                int cnt = rem[c][k];
                if (cnt <= 0) continue;
                double prob = static_cast<double>(cnt) / total_hidden;
                JunqiBoard child = make_child(c, k);
                expected += prob * (-child_value(child));
            }
        }
        return expected;
    }

    // Star1：按概率降序（稳定），配 Star1 边界剪枝
    std::vector<std::array<int, 3>> outcomes;   // {color, kind, count}
    for (int c = 0; c < 2; ++c) {
        for (int k = 0; k < 12; ++k) {
            if (rem[c][k] > 0) outcomes.push_back({c, k, rem[c][k]});
        }
    }
    std::stable_sort(outcomes.begin(), outcomes.end(),
                     [](const std::array<int, 3>& a, const std::array<int, 3>& b) {
                         return a[2] > b[2];
                     });

    constexpr double v_max = 600.0;
    constexpr double v_min = -600.0;
    double expected_value = 0.0;
    double remaining_prob = 1.0;

    for (const auto& o : outcomes) {
        double prob = static_cast<double>(o[2]) / total_hidden;
        if (expected_value + remaining_prob * v_max <= alpha) {
            stats.star1_cutoffs++;
            return alpha;
        }
        if (expected_value + remaining_prob * v_min >= beta) {
            stats.star1_cutoffs++;
            return beta;
        }
        JunqiBoard child = make_child(o[0], o[1]);
        double v = -negamax_(child, depth - 1, ply_depth + 1, -EXPERT_WIN_SCORE, EXPERT_WIN_SCORE);
        expected_value += prob * v;
        remaining_prob -= prob;
    }
    return expected_value;
}

// ---------------------------------------------------------------- Negamax

double ExpertSearch::negamax_(JunqiBoard& b, int depth, int ply_depth,
                              double alpha, double beta) {
    stats.nodes++;

    if ((stats.nodes & 511) == 0 && has_deadline_ &&
        std::chrono::steady_clock::now().time_since_epoch() >= deadline_) {
        stopped = true;
        return alpha;
    }

    if (b.is_terminal()) {
        if (b.winner == -1) return 0.0;
        double win = EXPERT_WIN_SCORE - b.ply;
        return (b.winner == static_cast<int>(b.turn)) ? win : -win;
    }

    if (depth <= 0) return _qsearch(b, alpha, beta, qsearch_depth);

    uint64_t zk = b.compute_zobrist();
    if (in_path(zk)) return 0.0;

    bool has_tt_move = false;
    Action tt_move{};
    double cut = 0.0;
    bool cut_ok = tt_lookup(zk, depth, alpha, beta, &has_tt_move, &tt_move, &cut);
    if (cut_ok && ply_depth > 0) return cut;

    std::vector<Action> acts = b.legal_actions();
    if (acts.empty()) return -(EXPERT_WIN_SCORE - b.ply);

    OrderCtx ctx;
    ctx.has_tt_move = has_tt_move;
    ctx.tt_move = tt_move;
    ctx.ply_depth = ply_depth;
    std::vector<Action> ordered = order_actions(b, acts, ctx);

    double best_score = NEG_INF;
    Action best_act = ordered[0];
    int flag = TT_FLAG_UPPER_BOUND;

    path_.push_back(zk);
    for (size_t i = 0; i < ordered.size(); ++i) {
        if (stopped) break;
        const Action a = ordered[i];
        double score;
        if (a.kind == ActionKind::FLIP) {
            score = chance_flip_(b, a.frm, depth, ply_depth, alpha, beta);
        } else {
            JunqiBoard child = b;
            apply_expert(child, a);
            if (i == 0) {
                score = -negamax_(child, depth - 1, ply_depth + 1, -beta, -alpha);
            } else {
                score = -negamax_(child, depth - 1, ply_depth + 1, -alpha - 1, -alpha);
                if (alpha < score && score < beta && !stopped) {
                    stats.pvs_researches++;
                    score = -negamax_(child, depth - 1, ply_depth + 1, -beta, -score);
                }
            }
        }
        if (stopped) break;

        if (score > best_score) {
            best_score = score;
            best_act = a;
        }
        if (score > alpha) {
            alpha = score;
            flag = TT_FLAG_EXACT;
        }
        if (alpha >= beta) {
            flag = TT_FLAG_LOWER_BOUND;
            if (a.kind == ActionKind::MOVE) update_cutoff_heuristics(a, ply_depth, depth);
            break;
        }
    }
    path_.pop_back();

    if (!stopped) tt_store(zk, depth, best_score, flag, best_act);
    return best_score;
}

// ---------------------------------------------------------------- 顶层入口

double ExpertSearch::negamax_blob(const std::string& blob, int depth, int ply_depth,
                                  double alpha, double beta) {
    JunqiBoard b = board_from_blob(blob);
    path_.clear();
    return negamax_(b, depth, ply_depth, alpha, beta);
}

double ExpertSearch::chance_flip_blob(const std::string& blob, int pos, int depth,
                                      int ply_depth, double alpha, double beta) {
    JunqiBoard b = board_from_blob(blob);
    path_.clear();
    return chance_flip_(b, static_cast<uint8_t>(pos), depth, ply_depth, alpha, beta);
}

}  // namespace junqi
