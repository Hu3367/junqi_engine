#include "apk_engine.h"
#include <algorithm>
#include <chrono>

namespace junqi {

ApkSearchEngine::ApkSearchEngine(size_t tt_size_power, std::optional<uint64_t> seed)
    : rng_(seed.value_or(123456789ULL)) {
    size_t table_size = 1ULL << tt_size_power;
    tt_mask_ = table_size - 1;
    tt_table_.resize(table_size);
    clear_tt();
}

void ApkSearchEngine::clear_tt() {
    for (auto& entry : tt_table_) {
        entry.first = 0;
        entry.second = TTEntry{};
    }
}

TTEntry* ApkSearchEngine::tt_probe(uint64_t hash) {
    size_t idx = hash & tt_mask_;
    if (tt_table_[idx].first == hash) {
        return &tt_table_[idx].second;
    }
    return nullptr;
}

void ApkSearchEngine::tt_store(uint64_t hash, int depth, TTFlag flag, double score, const std::optional<Action>& best_act) {
    size_t idx = hash & tt_mask_;
    tt_table_[idx].first = hash;
    tt_table_[idx].second = TTEntry{depth, flag, score, best_act};
}

void ApkSearchEngine::order_root_moves(
    std::vector<Action>& moves,
    const JunqiBoard& board,
    const std::optional<Action>& pv_move
) {
    auto move_priority = [&](const Action& a) -> double {
        if (pv_move.has_value() && a == pv_move.value()) {
            return 1000000.0;
        }
        double p = 0.0;
        const Piece& target = board.cells[a.to];
        if (!target.is_empty() && target.revealed) {
            p += 100000.0 + get_apk_piece_value(target.rank);
        } else if (is_camp_idx(a.to) && !is_camp_idx(a.frm)) {
            p += 50000.0;
        }
        return p;
    };

    std::sort(moves.begin(), moves.end(), [&](const Action& a, const Action& b) {
        return move_priority(a) > move_priority(b);
    });
}

void ApkSearchEngine::order_moves(
    std::vector<Action>& moves,
    const JunqiBoard& board,
    const std::optional<Action>& tt_move
) {
    auto move_priority = [&](const Action& a) -> double {
        if (tt_move.has_value() && a == tt_move.value()) {
            return 1000000.0;
        }
        double p = 0.0;
        const Piece& target = board.cells[a.to];
        if (!target.is_empty() && target.revealed) {
            p += 100000.0 + get_apk_piece_value(target.rank);
        } else if (is_camp_idx(a.to) && !is_camp_idx(a.frm)) {
            p += 50000.0;
        }
        return p;
    };

    std::sort(moves.begin(), moves.end(), [&](const Action& a, const Action& b) {
        return move_priority(a) > move_priority(b);
    });
}

std::pair<Action, double> ApkSearchEngine::search(
    const JunqiBoard& board,
    const std::string& level,
    int depth_override,
    int time_limit_ms_override,
    int qsearch_depth_override
) {
    LevelSpec spec = get_apk_level_spec(level);
    int max_depth = (depth_override > 0) ? depth_override : spec.depth;
    int time_limit_ms = (time_limit_ms_override >= 0) ? time_limit_ms_override : spec.time_limit_ms;
    int qdepth = (qsearch_depth_override > 0) ? qsearch_depth_override : spec.qsearch_depth;
    double jitter_range = spec.jitter;

    stats_ = ApkSearchStats{};
    auto start_time = std::chrono::steady_clock::now();

    auto acts = board.legal_actions();
    if (acts.empty() || board.is_terminal()) {
        return {Action{}, 0.0};
    }
    if (acts.size() == 1) {
        return {acts[0], 0.0};
    }

    Color my_color = board.my_color();

    // 1. 根节点走法分类：走棋 (moves) 与 翻暗棋 (flips)
    std::vector<Action> moves;
    std::vector<Action> flips;
    for (const auto& a : acts) {
        if (a.kind == ActionKind::MOVE) moves.push_back(a);
        else flips.push_back(a);
    }

    // 2. 翻暗棋在根节点由启发式打分 (0x5a3c0)
    std::uniform_real_distribution<double> dist(-jitter_range, jitter_range);
    std::vector<std::pair<Action, double>> flip_scores;
    for (const auto& f : flips) {
        double f_score = eval_apk_flip_root(board, f, my_color);
        if (jitter_range > 0) {
            f_score += dist(rng_);
        }
        flip_scores.push_back({f, f_score});
    }

    // 3. 明子走法进入 Alpha-Beta + QSearch 树搜索
    std::optional<Action> best_move;
    double best_move_score = -999999.0;
    std::vector<std::pair<Action, double>> pv_move_scores;

    if (!moves.empty()) {
        JunqiBoard root_copy = board;

        // 迭代加深搜索 (IDS, 对齐 0x5ac3a)
        for (int cur_depth = 1; cur_depth <= max_depth; ++cur_depth) {
            std::optional<Action> cur_best_move;
            double cur_best_score = -999999.0;
            std::vector<std::pair<Action, double>> cur_scores;
            double alpha = -999999.0;
            double beta = 999999.0;

            order_root_moves(moves, root_copy, best_move);

            for (const auto& m : moves) {
                JunqiBoard nxt = root_copy;
                nxt.apply(m);
                stats_.nodes++;

                // 递归 Alpha-Beta 搜索（注意：树深层绝不生成翻棋）
                double score = -alpha_beta(
                    nxt, cur_depth - 1, -beta, -alpha, qdepth, start_time, time_limit_ms, my_color
                );

                cur_scores.push_back({m, score});

                if (score > cur_best_score) {
                    cur_best_score = score;
                    cur_best_move = m;
                }

                if (score > alpha) {
                    alpha = score;
                }

                // 耗时检查
                auto elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start_time
                ).count();
                if (time_limit_ms > 0 && elapsed_ms >= time_limit_ms) {
                    break;
                }
            }

            auto elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start_time
            ).count();

            if (cur_best_move.has_value()) {
                best_move = cur_best_move;
                best_move_score = cur_best_score;
                pv_move_scores = cur_scores;
                stats_.depth_reached = cur_depth;
            }

            // IDS 25% 提前跳出机制 (对齐 0x5ac3a: r2 > r3 asr #2)
            if (time_limit_ms > 0 && elapsed_ms >= (time_limit_ms * 0.25)) {
                break;
            }
        }
    }

    // 4. 根节点走法仲裁：明子最优走法 vs 翻棋最优走法
    std::vector<std::pair<Action, double>> all_candidate_scores;

    for (const auto& [m, sc] : pv_move_scores) {
        double adj_sc = sc;
        if (jitter_range > 0) {
            adj_sc += dist(rng_);
        }
        all_candidate_scores.push_back({m, adj_sc});
    }

    for (const auto& [f, sc] : flip_scores) {
        all_candidate_scores.push_back({f, sc});
    }

    std::sort(all_candidate_scores.begin(), all_candidate_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    stats_.root_scores = all_candidate_scores;
    stats_.time_spent_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start_time
    ).count();

    if (!all_candidate_scores.empty()) {
        return all_candidate_scores[0];
    }

    return {acts[0], 0.0};
}

double ApkSearchEngine::alpha_beta(
    JunqiBoard& board,
    int depth,
    double alpha,
    double beta,
    int qdepth,
    const std::chrono::steady_clock::time_point& start_time,
    int time_limit_ms,
    Color root_color
) {
    // 1. 终局检查
    if (board.is_terminal()) {
        int w = board.winner;
        if (w == -1 || w == -2) return 0.0;
        Color my_c = board.my_color();
        double base_win = (board.win_reason == "flag") ? 500000.0 : 400000.0;
        return (board.seat_color[w] == my_c) ? (base_win + depth * 1000.0) : (-base_win - depth * 1000.0);
    }

    // 2. 叶子节点转入静态搜索 (QSearch)
    if (depth <= 0) {
        return qsearch(board, alpha, beta, qdepth, root_color);
    }

    // 3. 超时检查
    if (time_limit_ms > 0) {
        auto elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_time
        ).count();
        if (elapsed_ms >= time_limit_ms) {
            return eval_apk_pure(board, board.my_color());
        }
    }

    // 4. 置换表 (TT) 探测
    uint64_t h = board.compute_zobrist();
    TTEntry* tt_entry = tt_probe(h);
    std::optional<Action> tt_move;
    if (tt_entry != nullptr) {
        tt_move = tt_entry->best_move;
        if (tt_entry->depth >= depth) {
            stats_.tt_hits++;
            if (tt_entry->flag == TTFlag::EXACT) {
                return tt_entry->score;
            } else if (tt_entry->flag == TTFlag::LOWER_BOUND && tt_entry->score >= beta) {
                return tt_entry->score;
            } else if (tt_entry->flag == TTFlag::UPPER_BOUND && tt_entry->score <= alpha) {
                return tt_entry->score;
            }
        }
    }

    // 5. 生成走法（严格遵循 0x59e80：树深层只搜明子，绝不生成翻暗棋）
    auto legal_acts = board.legal_actions();
    std::vector<Action> moves;
    for (const auto& a : legal_acts) {
        if (a.kind == ActionKind::MOVE) moves.push_back(a);
    }

    if (moves.empty()) {
        return eval_apk_pure(board, board.my_color());
    }

    order_moves(moves, board, tt_move);

    double best_score = -999999.0;
    std::optional<Action> best_act;
    double orig_alpha = alpha;

    // 6. PVS 循环
    for (size_t i = 0; i < moves.size(); ++i) {
        const Action& m = moves[i];
        JunqiBoard nxt = board;
        nxt.apply(m);
        stats_.nodes++;

        double score = 0.0;
        if (i == 0) {
            score = -alpha_beta(nxt, depth - 1, -beta, -alpha, qdepth, start_time, time_limit_ms, root_color);
        } else {
            score = -alpha_beta(nxt, depth - 1, -alpha - 1.0, -alpha, qdepth, start_time, time_limit_ms, root_color);
            if (score > alpha && score < beta) {
                score = -alpha_beta(nxt, depth - 1, -beta, -score, qdepth, start_time, time_limit_ms, root_color);
            }
        }

        if (score > best_score) {
            best_score = score;
            best_act = m;
        }

        if (score > alpha) {
            alpha = score;
        }

        if (alpha >= beta) {
            break; // Beta 剪枝
        }
    }

    // 行营驻守机制：翻棋对局中驻营子力无需强行出营送死，若全盘走法均劣于驻守则保持阵地
    Color my_c = board.my_color();
    bool has_camp_piece = false;
    for (uint8_t c_idx : CAMPS) {
        const Piece& cp = board.cells[c_idx];
        if (!cp.is_empty() && cp.revealed && cp.color == my_c) {
            has_camp_piece = true;
            break;
        }
    }
    if (has_camp_piece) {
        double stand_pat = eval_apk_pure(board, my_c);
        if (best_score < stand_pat) {
            best_score = stand_pat;
        }
    }

    // 7. 存入置换表
    TTFlag flag = TTFlag::EXACT;
    if (best_score <= orig_alpha) {
        flag = TTFlag::UPPER_BOUND;
    } else if (best_score >= beta) {
        flag = TTFlag::LOWER_BOUND;
    }
    tt_store(h, depth, flag, best_score, best_act);

    return best_score;
}

double ApkSearchEngine::qsearch(
    JunqiBoard& board,
    double alpha,
    double beta,
    int qdepth,
    Color root_color
) {
    stats_.qnodes++;
    Color my_c = board.my_color();

    if (board.is_terminal()) {
        int w = board.winner;
        if (w == -1 || w == -2) return 0.0;
        return (board.seat_color[w] == my_c) ? 300000.0 : -300000.0;
    }

    double stand_pat = eval_apk_pure(board, my_c);

    if (qdepth <= 0 || stand_pat >= beta) {
        return stand_pat;
    }

    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    // Delta 剪枝：即使吃掉司令 (2560) 仍无法超越 alpha，直接剪除
    if (stand_pat + 2560.0 < alpha) {
        return alpha;
    }

    // 严格只生成吃子动作 (Captures Only)
    auto acts = board.legal_actions();
    std::vector<Action> captures;
    for (const auto& a : acts) {
        if (a.kind == ActionKind::MOVE) {
            const Piece& target = board.cells[a.to];
            if (!target.is_empty() && target.revealed) {
                captures.push_back(a);
            }
        }
    }

    if (captures.empty()) {
        return stand_pat;
    }

    // 吃子排序 (MVV-LVA)
    std::sort(captures.begin(), captures.end(), [&](const Action& a, const Action& b) {
        return get_apk_piece_value(board.cells[a.to].rank) > get_apk_piece_value(board.cells[b.to].rank);
    });

    for (const auto& cap : captures) {
        const Piece& victim = board.cells[cap.to];
        double victim_val = get_apk_piece_value(victim.rank);

        // 局部 Delta 剪枝
        if (stand_pat + victim_val + 200.0 < alpha) {
            continue;
        }

        JunqiBoard nxt = board;
        nxt.apply(cap);

        double score = -qsearch(nxt, -beta, -alpha, qdepth - 1, root_color);

        if (score >= beta) {
            return beta;
        }

        if (score > alpha) {
            alpha = score;
        }
    }

    return alpha;
}

} // namespace junqi
