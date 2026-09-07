#include "rules.h"
#include "constants.h"
#include <queue>
#include <set>
#include <algorithm>

namespace junqi {

// 全局公路和铁路邻接表单例
static std::array<CellNeighbors, NUM_CELLS> s_road_neighbors;
static std::array<CellNeighbors, NUM_CELLS> s_rail_neighbors;
static bool s_tables_initialized = false;

static void init_tables_if_needed() {
    if (s_tables_initialized) return;

    for (int idx = 0; idx < NUM_CELLS; ++idx) {
        s_road_neighbors[idx].count = 0;
        s_rail_neighbors[idx].count = 0;
    }

    // 1. 公路网正交边
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int u = pos_to_idx(r, c);
            if (r + 1 < ROWS) {
                int v = pos_to_idx(r + 1, c);
                if (!is_cross_blocked(u, v)) {
                    s_road_neighbors[u].neighbors[s_road_neighbors[u].count++] = v;
                    s_road_neighbors[v].neighbors[s_road_neighbors[v].count++] = u;
                }
            }
            if (c + 1 < COLS) {
                int v = pos_to_idx(r, c + 1);
                s_road_neighbors[u].neighbors[s_road_neighbors[u].count++] = v;
                s_road_neighbors[v].neighbors[s_road_neighbors[v].count++] = u;
            }
        }
    }

    // 2. 行营 4 条斜向通道
    for (uint8_t camp_idx : CAMPS) {
        int cr = camp_idx / COLS;
        int cc = camp_idx % COLS;
        for (int dr : {-1, 1}) {
            for (int dc : {-1, 1}) {
                int nr = cr + dr;
                int nc = cc + dc;
                if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                    int v = pos_to_idx(nr, nc);
                    // 确保不重复添加
                    bool exists = false;
                    for (int i = 0; i < s_road_neighbors[camp_idx].count; ++i) {
                        if (s_road_neighbors[camp_idx].neighbors[i] == v) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists && s_road_neighbors[camp_idx].count < 8 && s_road_neighbors[v].count < 8) {
                        s_road_neighbors[camp_idx].neighbors[s_road_neighbors[camp_idx].count++] = v;
                        s_road_neighbors[v].neighbors[s_road_neighbors[v].count++] = camp_idx;
                    }
                }
            }
        }
    }

    // 3. 铁路网正交单步邻接
    for (int idx = 0; idx < NUM_CELLS; ++idx) {
        if (!is_rail_idx(idx)) continue;
        int r = idx / COLS;
        int c = idx % COLS;
        for (int i = 0; i < s_road_neighbors[idx].count; ++i) {
            uint8_t n = s_road_neighbors[idx].neighbors[i];
            if (is_rail_idx(n) && !is_cross_blocked(idx, n)) {
                int nr = n / COLS;
                int nc = n % COLS;
                if (std::abs(nr - r) + std::abs(nc - c) == 1) {
                    s_rail_neighbors[idx].neighbors[s_rail_neighbors[idx].count++] = n;
                }
            }
        }
    }

    s_tables_initialized = true;
}

const std::array<CellNeighbors, NUM_CELLS>& get_road_neighbors() {
    init_tables_if_needed();
    return s_road_neighbors;
}

const std::array<CellNeighbors, NUM_CELLS>& get_rail_neighbors() {
    init_tables_if_needed();
    return s_rail_neighbors;
}

BattleResult resolve_battle(Rank attacker, Rank defender) {
    if (defender == Rank::QI) {
        return BattleResult::ATTACKER_WINS;
    }
    if (attacker == Rank::ZHA || defender == Rank::ZHA) {
        return BattleResult::BOTH_DIE;
    }
    if (defender == Rank::LEI) {
        return (attacker == Rank::GONG) ? BattleResult::ATTACKER_WINS : BattleResult::DEFENDER_WINS;
    }
    if (attacker == defender) {
        return BattleResult::BOTH_DIE;
    }
    return (attacker > defender) ? BattleResult::ATTACKER_WINS : BattleResult::DEFENDER_WINS;
}

// 四个直行方向
static const int DIRS_R[4] = {-1, 1, 0, 0};
static const int DIRS_C[4] = {0, 0, 1, -1};

void generate_rail_slides(
    const JunqiBoard& board,
    uint8_t start,
    Rank attacker_rank,
    bool flag_ok,
    std::vector<Action>& out_actions
) {
    int start_r = start / COLS;
    int start_c = start % COLS;

    for (int d = 0; d < 4; ++d) {
        int dr = DIRS_R[d];
        int dc = DIRS_C[d];
        int cr = start_r;
        int cc = start_c;

        while (true) {
            int nr = cr + dr;
            int nc = cc + dc;
            if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) break;
            uint8_t n_idx = pos_to_idx(nr, nc);
            uint8_t cur_idx = pos_to_idx(cr, cc);

            if (!is_rail_idx(n_idx) || is_cross_blocked(cur_idx, n_idx)) {
                break;
            }

            const Piece& target = board.cells[n_idx];
            if (!target.is_empty()) {
                if (board.is_attackable(attacker_rank, target, n_idx, flag_ok)) {
                    out_actions.push_back(Action::make_move(start, n_idx));
                }
                break; // 被棋子阻挡，停止该方向滑行
            }

            out_actions.push_back(Action::make_move(start, n_idx));
            cr = nr;
            cc = nc;
        }
    }
}

void generate_engineer_flights(
    const JunqiBoard& board,
    uint8_t start,
    Rank attacker_rank,
    bool flag_ok,
    bool can_fly_over_pieces,
    std::vector<Action>& out_actions
) {
    init_tables_if_needed();

    // 状态队列：(pos, dir)
    // 0: N, 1: S, 2: E, 3: W
    struct QueueNode {
        uint8_t pos;
        int dir;
    };

    std::queue<QueueNode> q;
    // 记录是否访问过 (pos * 4 + dir)
    std::array<bool, NUM_CELLS * 4> seen{};

    for (int d = 0; d < 4; ++d) {
        q.push({start, d});
    }

    while (!q.empty()) {
        QueueNode node = q.front();
        q.pop();

        int state_id = node.pos * 4 + node.dir;
        if (seen[state_id]) continue;
        seen[state_id] = true;

        int cr = node.pos / COLS;
        int cc = node.pos % COLS;
        int nr = cr + DIRS_R[node.dir];
        int nc = cc + DIRS_C[node.dir];

        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) continue;
        uint8_t n_idx = pos_to_idx(nr, nc);

        if (!is_rail_idx(n_idx) || is_cross_blocked(node.pos, n_idx)) continue;

        const Piece& target = board.cells[n_idx];
        if (!target.is_empty()) {
            if (board.is_attackable(attacker_rank, target, n_idx, flag_ok)) {
                out_actions.push_back(Action::make_move(start, n_idx));
            }
            if (!can_fly_over_pieces) {
                continue; // 阻挡不可穿透
            }
        } else {
            out_actions.push_back(Action::make_move(start, n_idx));
        }

        // 继续同向移动
        q.push({n_idx, node.dir});

        // 尝试转弯 (拐入其他方向)
        for (int nd = 0; nd < 4; ++nd) {
            if (nd != node.dir) {
                q.push({n_idx, nd});
            }
        }
    }
}

std::vector<Action> generate_all_legal_actions(const JunqiBoard& board) {
    init_tables_if_needed();
    std::vector<Action> acts;
    acts.reserve(64);

    Color my = board.my_color();
    auto hidden = board.hidden_positions();

    // 1. 翻暗棋动作永远可选
    for (uint8_t pos : hidden) {
        acts.push_back(Action::make_flip(pos));
    }

    // 首翻前双方颜色未定，无明子可走
    if (my == Color::NONE) {
        return acts;
    }

    bool flag_ok = board.flag_attackable();

    // 2. 遍历己方明子生成移动
    for (uint8_t pos = 0; pos < NUM_CELLS; ++pos) {
        const Piece& pc = board.cells[pos];
        if (pc.is_empty() || !pc.revealed || pc.color != my) continue;
        if (pc.rank == Rank::LEI || pc.rank == Rank::QI) continue;
        if (board.cfg.hq_locks_pieces && is_hq_idx(pos)) continue;

        // 公路一步（含行营斜道）
        const auto& nb = s_road_neighbors[pos];
        for (int i = 0; i < nb.count; ++i) {
            uint8_t np = nb.neighbors[i];
            const Piece& target = board.cells[np];
            if (target.is_empty() || board.is_attackable(pc.rank, target, np, flag_ok)) {
                acts.push_back(Action::make_move(pos, np));
            }
        }

        // 铁路移动
        if (is_rail_idx(pos)) {
            if (pc.rank == Rank::GONG && board.cfg.engineer_rail_turns) {
                generate_engineer_flights(board, pos, pc.rank, flag_ok, board.cfg.engineer_can_fly_over_pieces, acts);
            } else {
                generate_rail_slides(board, pos, pc.rank, flag_ok, acts);
            }
        }
    }

    // 去除重复生成的动作（例如同一移动由公路与铁路同时生成）
    std::sort(acts.begin(), acts.end());
    acts.erase(std::unique(acts.begin(), acts.end()), acts.end());

    return acts;
}

} // namespace junqi
