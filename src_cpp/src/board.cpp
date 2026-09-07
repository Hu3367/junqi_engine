#include "board.h"
#include "rules.h"
#include "zobrist.h"
#include <algorithm>

namespace junqi {

bool JunqiBoard::flag_attackable() const {
    if (cfg.flag_needs_all_flipped && remaining_hidden_count() > 0) {
        return false;
    }
    if (cfg.flag_needs_mines_cleared) {
        Color enemy = other_color(my_color());
        int dead_mines = 0;
        for (const auto& p : dead) {
            if (p.color == enemy && p.rank == Rank::LEI) {
                dead_mines++;
            }
        }
        if (dead_mines < 3) { // 对方 3 颗地雷必须挖光
            return false;
        }
    }
    return true;
}

bool JunqiBoard::is_attackable(Rank attacker, const Piece& target, uint8_t tpos, bool flag_ok) const {
    if (!target.revealed) {
        return false; // 暗子不可攻击
    }
    if (target.color == my_color()) {
        return false; // 友军不可吃
    }
    if (is_camp_idx(tpos)) {
        return false; // 行营内不可攻击
    }
    if (target.rank == Rank::QI) {
        if (!flag_ok) return false;
        if (cfg.flag_gong_only && attacker != Rank::GONG) return false;
    }
    if (!cfg.allow_suicide_attack) {
        if (resolve_battle(attacker, target.rank) == BattleResult::DEFENDER_WINS) {
            return false; // 禁止自杀攻击
        }
    }
    return true;
}

BattleResult JunqiBoard::battle(Rank attacker, Rank defender) const {
    return resolve_battle(attacker, defender);
}

std::vector<Action> JunqiBoard::legal_actions() const {
    return generate_all_legal_actions(*this);
}

void JunqiBoard::apply(const Action& act) {
    quiet++;
    ply++;

    if (act.kind == ActionKind::FLIP) {
        cells[act.frm].revealed = true;
        if (!first_flip_done) {
            seat_color[turn] = cells[act.frm].color;
            seat_color[1 - turn] = other_color(cells[act.frm].color);
            first_flip_done = true;
        }
    } else { // MOVE
        Piece mover = cells[act.frm];
        cells[act.frm] = Piece{}; // 清空原位

        Piece target = cells[act.to];
        if (target.is_empty()) {
            cells[act.to] = mover;
        } else {
            quiet = 0; // 吃子清零
            BattleResult res = resolve_battle(mover.rank, target.rank);
            if (res == BattleResult::ATTACKER_WINS) {
                dead.push_back(target);
                cells[act.to] = mover;
                if (target.rank == Rank::QI) {
                    winner = turn;
                    win_reason = "flag";
                }
            } else if (res == BattleResult::BOTH_DIE) {
                dead.push_back(mover);
                dead.push_back(target);
                cells[act.to] = Piece{}; // 两败俱伤
            } else { // DEFENDER_WINS
                dead.push_back(mover);
            }
        }
    }

    // 检查和棋条件
    if (winner == -2) {
        if (cfg.no_capture_draw_plies > 0 && quiet >= cfg.no_capture_draw_plies) {
            winner = -1;
            win_reason = "no_capture";
        } else if (cfg.max_plies > 0 && ply >= cfg.max_plies) {
            winner = -1;
            win_reason = "max_plies";
        }
    }

    // 切换行动方
    turn = 1 - turn;

    // 困毙检查 (仅当胜负未定时)
    if (winner == -2) {
        auto acts = legal_actions();
        if (acts.empty()) {
            winner = 1 - turn; // 无棋可走判负，对手获胜
            win_reason = "immobilized";
        }
    }
}

uint64_t JunqiBoard::compute_zobrist() const {
    return compute_board_zobrist(*this);
}

} // namespace junqi
