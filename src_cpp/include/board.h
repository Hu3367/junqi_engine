#pragma once

#include <array>
#include <vector>
#include <string>
#include <optional>
#include "types.h"
#include "constants.h"
#include "config.h"

namespace junqi {

class JunqiBoard {
public:
    std::array<Piece, NUM_CELLS> cells{};
    std::vector<Piece> dead{};
    std::array<Color, 2> seat_color{Color::NONE, Color::NONE};
    uint8_t turn{0};
    bool first_flip_done{false};
    int ply{0};
    int quiet{0};
    int winner{-2}; // -2: ongoing, -1: draw, 0: seat 0, 1: seat 1
    std::string win_reason{};
    RuleConfig cfg{};

    JunqiBoard() = default;

    Color my_color(int seat = -1) const {
        int s = (seat == -1) ? turn : seat;
        return seat_color[s];
    }

    bool is_terminal() const {
        return winner != -2;
    }

    std::vector<uint8_t> hidden_positions() const {
        std::vector<uint8_t> res;
        for (uint8_t i = 0; i < NUM_CELLS; ++i) {
            if (!cells[i].is_empty() && !cells[i].revealed) {
                res.push_back(i);
            }
        }
        return res;
    }

    int remaining_hidden_count() const {
        int cnt = 0;
        for (uint8_t i = 0; i < NUM_CELLS; ++i) {
            if (!cells[i].is_empty() && !cells[i].revealed) {
                cnt++;
            }
        }
        return cnt;
    }

    bool flag_attackable() const;

    bool is_attackable(Rank attacker, const Piece& target, uint8_t tpos, bool flag_ok) const;

    BattleResult battle(Rank attacker, Rank defender) const;

    std::vector<Action> legal_actions() const;

    void apply(const Action& act);

    JunqiBoard clone() const {
        return *this;
    }

    uint64_t compute_zobrist() const;
};

} // namespace junqi
