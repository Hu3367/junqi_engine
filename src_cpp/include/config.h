#pragma once

#include <cstdint>
#include <string>

namespace junqi {

struct RuleConfig {
    bool flag_needs_mines_cleared{true};
    bool flag_needs_all_flipped{false};
    bool flag_gong_only{true};
    bool allow_suicide_attack{false};
    bool hq_locks_pieces{false};
    bool engineer_rail_turns{true};
    bool engineer_can_fly_over_pieces{false};
    int no_capture_draw_plies{70};
    int max_plies{1000};
    int repetition_draw_count{3};
};

struct LevelSpec {
    int depth{4};
    int time_limit_ms{1000};
    int qsearch_depth{16};
    double jitter{0.5};
};

inline LevelSpec get_apk_level_spec(const std::string& level) {
    if (level == "beginner") {
        return {2, 100, 8, 30.0};
    } else if (level == "intermediate") {
        return {3, 300, 12, 10.0};
    } else { // "advanced"
        return {4, 1000, 16, 0.5};
    }
}

} // namespace junqi
