#include "zobrist.h"
#include "board.h"
#include <random>

namespace junqi {

ZobristTable::ZobristTable() {
    std::mt19937_64 rng(0x4A756E51695A6F62ULL); // "JunQiZob"

    for (int p = 0; p < NUM_CELLS; ++p) {
        for (int c = 0; c < 2; ++c) {
            for (int r = 0; r < 23; ++r) {
                for (int rev = 0; rev < 2; ++rev) {
                    piece_keys_[p][c][r][rev] = rng();
                }
            }
        }
    }

    turn_keys_[0] = rng();
    turn_keys_[1] = rng();

    seat_color_keys_[0] = rng(); // None
    seat_color_keys_[1] = rng(); // R-B
    seat_color_keys_[2] = rng(); // B-R
}

const ZobristTable& ZobristTable::instance() {
    static ZobristTable s_inst;
    return s_inst;
}

uint64_t ZobristTable::piece_key(uint8_t pos, Color c, Rank r, bool revealed) const {
    int c_idx = (c == Color::RED) ? 0 : 1;
    int r_idx = static_cast<int>(r);
    int rev_idx = revealed ? 1 : 0;
    return piece_keys_[pos][c_idx][r_idx][rev_idx];
}

uint64_t ZobristTable::turn_key(uint8_t turn) const {
    return turn_keys_[turn & 1];
}

uint64_t ZobristTable::seat_color_key(Color seat0, Color seat1) const {
    if (seat0 == Color::RED) return seat_color_keys_[1];
    if (seat0 == Color::BLUE) return seat_color_keys_[2];
    return seat_color_keys_[0];
}

uint64_t compute_board_zobrist(const JunqiBoard& board) {
    const auto& zt = ZobristTable::instance();
    uint64_t h = 0;

    for (uint8_t pos = 0; pos < NUM_CELLS; ++pos) {
        const Piece& pc = board.cells[pos];
        if (pc.is_empty()) continue;
        if (pc.revealed) {
            h ^= zt.piece_key(pos, pc.color, pc.rank, true);
        } else {
            // 暗子公共视野只记暗子占位（严防公共视野透视真实暗子）
            h ^= zt.piece_key(pos, Color::RED, Rank::EMPTY, false);
        }
    }

    h ^= zt.turn_key(board.turn);
    h ^= zt.seat_color_key(board.seat_color[0], board.seat_color[1]);

    return h;
}

} // namespace junqi
