#pragma once

#include <vector>
#include <unordered_map>
#include <random>
#include <chrono>
#include <optional>
#include "board.h"
#include "eval_apk.h"

namespace junqi {

struct ApkSearchStats {
    int nodes{0};
    int qnodes{0};
    int tt_hits{0};
    int depth_reached{0};
    double time_spent_ms{0.0};
    std::vector<std::pair<Action, double>> root_scores{};
};

enum class TTFlag : uint8_t {
    EXACT = 1,
    LOWER_BOUND = 2,
    UPPER_BOUND = 3
};

struct TTEntry {
    int depth{0};
    TTFlag flag{TTFlag::EXACT};
    double score{0.0};
    std::optional<Action> best_move{};
};

class ApkSearchEngine {
public:
    explicit ApkSearchEngine(size_t tt_size_power = 18, std::optional<uint64_t> seed = std::nullopt);

    void clear_tt();

    std::pair<Action, double> search(
        const JunqiBoard& board,
        const std::string& level = "advanced",
        int depth_override = -1,
        int time_limit_ms_override = -1,
        int qsearch_depth_override = -1
    );

    const ApkSearchStats& get_stats() const { return stats_; }

private:
    std::vector<std::pair<uint64_t, TTEntry>> tt_table_;
    size_t tt_mask_{0};
    ApkSearchStats stats_{};
    std::mt19937_64 rng_;

    double alpha_beta(
        JunqiBoard& board,
        int depth,
        double alpha,
        double beta,
        int qdepth,
        const std::chrono::steady_clock::time_point& start_time,
        int time_limit_ms,
        Color root_color
    );

    double qsearch(
        JunqiBoard& board,
        double alpha,
        double beta,
        int qdepth,
        Color root_color
    );

    void order_root_moves(
        std::vector<Action>& moves,
        const JunqiBoard& board,
        const std::optional<Action>& pv_move
    );

    void order_moves(
        std::vector<Action>& moves,
        const JunqiBoard& board,
        const std::optional<Action>& tt_move
    );

    TTEntry* tt_probe(uint64_t hash);
    void tt_store(uint64_t hash, int depth, TTFlag flag, double score, const std::optional<Action>& best_act);
};

} // namespace junqi
