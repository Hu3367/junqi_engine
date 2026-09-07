#pragma once

#include <cstdint>
#include "types.h"

namespace junqi {

class ZobristTable {
public:
    static const ZobristTable& instance();

    uint64_t piece_key(uint8_t pos, Color c, Rank r, bool revealed) const;
    uint64_t turn_key(uint8_t turn) const;
    uint64_t seat_color_key(Color seat0, Color seat1) const;

private:
    ZobristTable();
    uint64_t piece_keys_[NUM_CELLS][2][23][2]{}; // pos, color, rank, revealed
    uint64_t turn_keys_[2]{};
    uint64_t seat_color_keys_[3]{}; // None, R-B, B-R
};

uint64_t compute_board_zobrist(const class JunqiBoard& board);

} // namespace junqi
