#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace junqi {

constexpr int ROWS = 12;
constexpr int COLS = 5;
constexpr int NUM_CELLS = 60;

constexpr uint8_t pos_to_idx(int r, int c) {
    return static_cast<uint8_t>(r * COLS + c);
}

inline std::pair<int, int> idx_to_pos(int idx) {
    return {idx / COLS, idx % COLS};
}

enum class Color : uint8_t {
    RED = 0,
    BLUE = 1,
    NONE = 255
};

inline Color other_color(Color c) {
    if (c == Color::RED) return Color::BLUE;
    if (c == Color::BLUE) return Color::RED;
    return Color::NONE;
}

inline std::string color_to_str(Color c) {
    if (c == Color::RED) return "r";
    if (c == Color::BLUE) return "b";
    return "none";
}

inline Color str_to_color(const std::string& s) {
    if (s == "r" || s == "red" || s == "orange") return Color::RED;
    if (s == "b" || s == "blue" || s == "purple") return Color::BLUE;
    return Color::NONE;
}

enum class Rank : uint8_t {
    EMPTY = 0,
    GONG  = 5,   // 工兵
    PAI   = 6,   // 排长
    LIAN  = 7,   // 连长
    YING  = 8,   // 营长
    TUAN  = 9,   // 团长
    LV    = 10,  // 旅长
    SHI   = 11,  // 师长
    JUN   = 12,  // 军长
    SI    = 13,  // 司令
    ZHA   = 20,  // 炸弹
    LEI   = 21,  // 地雷
    QI    = 22   // 军旗
};

inline bool is_regular_rank(Rank r) {
    return static_cast<uint8_t>(r) >= 5 && static_cast<uint8_t>(r) <= 13;
}

struct Piece {
    Rank rank{Rank::EMPTY};
    Color color{Color::NONE};
    bool revealed{false};

    bool is_empty() const { return rank == Rank::EMPTY; }
};

enum class ActionKind : uint8_t {
    FLIP = 0,
    MOVE = 1
};

struct Action {
    ActionKind kind{ActionKind::FLIP};
    uint8_t frm{0};   // 0..59
    uint8_t to{0};    // 0..59 (for move)

    static Action make_flip(uint8_t pos) {
        Action a;
        a.kind = ActionKind::FLIP;
        a.frm = pos;
        a.to = pos;
        return a;
    }

    static Action make_move(uint8_t from_pos, uint8_t to_pos) {
        Action a;
        a.kind = ActionKind::MOVE;
        a.frm = from_pos;
        a.to = to_pos;
        return a;
    }

    bool operator==(const Action& o) const {
        if (kind != o.kind) return false;
        if (frm != o.frm) return false;
        if (kind == ActionKind::MOVE && to != o.to) return false;
        return true;
    }

    bool operator<(const Action& o) const {
        if (kind != o.kind) return kind < o.kind;
        if (frm != o.frm) return frm < o.frm;
        return to < o.to;
    }
};

enum class BattleResult : uint8_t {
    ATTACKER_WINS = 0,
    DEFENDER_WINS = 1,
    BOTH_DIE      = 2
};

} // namespace junqi
