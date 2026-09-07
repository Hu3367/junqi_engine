#pragma once

#include <vector>
#include "board.h"

namespace junqi {

BattleResult resolve_battle(Rank attacker, Rank defender);

void generate_rail_slides(
    const JunqiBoard& board,
    uint8_t start,
    Rank attacker_rank,
    bool flag_ok,
    std::vector<Action>& out_actions
);

void generate_engineer_flights(
    const JunqiBoard& board,
    uint8_t start,
    Rank attacker_rank,
    bool flag_ok,
    bool can_fly_over_pieces,
    std::vector<Action>& out_actions
);

std::vector<Action> generate_all_legal_actions(const JunqiBoard& board);

} // namespace junqi
