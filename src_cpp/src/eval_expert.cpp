#include "eval_expert.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <vector>

namespace junqi {

namespace {

// 战斗结算，等价于 junqi/rules.py::battle
enum : int { BR_ATTACKER = 0, BR_DEFENDER = 1, BR_BOTH = 2 };

inline int battle_res(int a, int d) {
    if (d == 22) return BR_ATTACKER;                          // 军旗
    if (a == 20 || d == 20) return BR_BOTH;                   // 炸弹
    if (d == 21) return (a == 5) ? BR_ATTACKER : BR_DEFENDER; // 地雷
    if (a == d) return BR_BOTH;
    return (a > d) ? BR_ATTACKER : BR_DEFENDER;
}

inline const CellNeighbors& nb_of(int idx) {
    return get_expert_road_neighbors()[idx];
}

// 复刻 Python round(x, 3)：十进制四舍五入 + **银行家进位**（half-to-even）。
// fortress_score 的软分用到了它，简单用 std::round 会在 .5 处与 Python 分歧。
double py_round3(double x) {
    double scaled = x * 1000.0;
    double fl = std::floor(scaled);
    double frac = scaled - fl;
    double res;
    if (frac > 0.5) {
        res = fl + 1.0;
    } else if (frac < 0.5) {
        res = fl;
    } else {
        res = (std::fmod(fl, 2.0) == 0.0) ? fl : (fl + 1.0);
    }
    return res / 1000.0;
}

}  // namespace

// ---------------------------------------------------------------- 构建

ExpertState make_expert_state(const std::string& board_bytes,
                              const std::string& dead_bytes,
                              int turn, int ply, int quiet,
                              int seat0, int seat1,
                              bool flag_needs_mines_cleared,
                              bool flag_gong_only, bool hq_locks_pieces,
                              int no_capture_draw_plies) {
    ExpertState st;
    st.turn = turn;
    st.ply = ply;
    st.quiet = quiet;
    st.seat_color[0] = seat0;
    st.seat_color[1] = seat1;
    st.flag_needs_mines_cleared = flag_needs_mines_cleared;
    st.flag_gong_only = flag_gong_only;
    st.hq_locks_pieces = hq_locks_pieces;
    st.no_capture_draw_plies = no_capture_draw_plies;

    st.rank.fill(0);
    st.color.fill(0);
    st.revealed.fill(false);
    st.occupied.fill(false);
    for (int i = 0; i < NUM_CELLS; ++i) {
        uint8_t code = static_cast<uint8_t>(board_bytes[static_cast<size_t>(i)]);
        uint8_t rv = code & 0x1F;
        if (rv == 0) continue;
        st.rank[i] = rv;
        st.color[i] = (code >> 5) & 1;
        st.revealed[i] = (code & 0x40) != 0;
        st.occupied[i] = true;
        if (!st.revealed[i]) {
            st.hidden_count++;
        } else {
            int k = rank_kind(rv);
            if (k >= 0) st.revealed_count[st.color[i]][k]++;
        }
    }

    for (size_t i = 0; i < dead_bytes.size(); ++i) {
        uint8_t code = static_cast<uint8_t>(dead_bytes[i]);
        uint8_t rv = code & 0x1F;
        if (rv == 0) continue;
        int c = (code >> 5) & 1;
        int k = rank_kind(rv);
        if (k >= 0) st.dead_count[c][k]++;
        st.total_dead++;
    }
    return st;
}

// ---------------------------------------------------------------- fortress_score

double fortress_score_cpp(const ExpertState& st, int seat) {
    int my = st.seat_color[seat];
    if (my < 0) return 0.0;
    int opp = 1 - my;

    // 1. 己方明军旗（idx 升序首个）
    int flag_pos = -1;
    for (int i = 0; i < NUM_CELLS; ++i) {
        if (st.occupied[i] && st.revealed[i] && st.color[i] == my && st.rank[i] == 22) {
            flag_pos = i;
            break;
        }
    }
    if (flag_pos < 0) return 0.0;

    // 2. 对方工兵是否全灭：= COMPOSITION − 阵亡（明子无论翻否都在盘上）
    int opp_engineers_alive = RANK_COMP[R_GONG] - st.dead_count[opp][R_GONG];

    // 永久墙
    bool perm_wall[NUM_CELLS];
    std::memset(perm_wall, 0, sizeof(perm_wall));
    for (int i = 0; i < NUM_CELLS; ++i) {
        if (!st.occupied[i] || !st.revealed[i]) continue;
        if (st.color[i] == opp && st.rank[i] == 21) {
            perm_wall[i] = true;
        } else if (st.color[i] == my && st.rank[i] == 21 && opp_engineers_alive == 0) {
            perm_wall[i] = true;
        }
    }

    // 3. 敌方威胁源：所有暗子 + 对方非雷非旗明子
    std::vector<int> sources;
    for (int i = 0; i < NUM_CELLS; ++i) {
        if (!st.occupied[i]) continue;
        if (!st.revealed[i] || (st.color[i] == opp && st.rank[i] != 21 && st.rank[i] != 22)) {
            sources.push_back(i);
        }
    }
    if (sources.empty()) return 1.0;

    // 4. 沿空格 BFS
    std::vector<char> vis(NUM_CELLS, 0);
    std::deque<int> q;
    for (int s : sources) {
        vis[s] = 1;
        q.push_back(s);
    }
    bool flag_reached = false;
    while (!q.empty()) {
        int cur = q.front();
        q.pop_front();
        if (cur == flag_pos) {
            flag_reached = true;
            break;
        }
        const CellNeighbors& nb = nb_of(cur);
        for (int k = 0; k < nb.count; ++k) {
            int nx = nb.neighbors[k];
            if (vis[nx]) continue;
            if (nx == flag_pos || !st.occupied[nx]) {
                vis[nx] = 1;
                q.push_back(nx);
            }
        }
    }
    if (!flag_reached) return 1.0;

    const CellNeighbors& fa = nb_of(flag_pos);
    if (fa.count == 0) return 0.0;
    int blocked = 0;
    for (int k = 0; k < fa.count; ++k) {
        int nx = fa.neighbors[k];
        if (perm_wall[nx]) {
            blocked++;
        } else if (st.occupied[nx] && st.revealed[nx] && st.color[nx] == my) {
            blocked++;
        }
    }
    return py_round3(0.3 * (static_cast<double>(blocked) / fa.count));
}

// ---------------------------------------------------------------- is_dead_draw

bool is_dead_draw_cpp(const ExpertState& st, bool ignore_quiet_limit) {
    int my = st.seat_color[st.turn];
    if (my < 0) return false;
    int opp = 1 - my;

    // 0. 规则层无吃子限步
    if (!ignore_quiet_limit && st.no_capture_draw_plies > 0 &&
        st.quiet >= st.no_capture_draw_plies) {
        return true;
    }
    if (st.hidden_count > 0) return false;

    int my_gong_alive = RANK_COMP[R_GONG] - st.dead_count[my][R_GONG];
    int opp_gong_alive = RANK_COMP[R_GONG] - st.dead_count[opp][R_GONG];

    int my_mines = 0, opp_mines = 0;
    std::vector<int> combat_my, combat_opp;
    for (int i = 0; i < NUM_CELLS; ++i) {
        if (!st.occupied[i] || !st.revealed[i]) continue;
        if (st.rank[i] == 21) {
            if (st.color[i] == my) my_mines++;
            else opp_mines++;
            continue;
        }
        if (st.rank[i] == 22) continue;
        if (st.color[i] == my) combat_my.push_back(i);
        else combat_opp.push_back(i);
    }

    // 原型 1: 双无工兵死锁
    bool my_can_flag = true, opp_can_flag = true;
    if (st.flag_gong_only) {
        if (my_gong_alive == 0) my_can_flag = false;
        if (opp_gong_alive == 0) opp_can_flag = false;
    }
    if (st.flag_needs_mines_cleared) {
        if (my_gong_alive == 0 && opp_mines > 0) my_can_flag = false;
        if (opp_gong_alive == 0 && my_mines > 0) opp_can_flag = false;
    }

    if (!my_can_flag && !opp_can_flag) {
        if (combat_my.empty() && combat_opp.empty()) return true;
        if (combat_my.empty() || combat_opp.empty()) return false;

        int max_rank_my = 0, max_rank_opp = 0;
        bool has_zha_my = false, has_zha_opp = false;
        for (int p : combat_my) {
            max_rank_my = std::max<int>(max_rank_my, st.rank[p]);
            if (st.rank[p] == 20) has_zha_my = true;
        }
        for (int p : combat_opp) {
            max_rank_opp = std::max<int>(max_rank_opp, st.rank[p]);
            if (st.rank[p] == 20) has_zha_opp = true;
        }

        bool can_kill_opp_max = (max_rank_my >= max_rank_opp) || has_zha_my;
        bool can_kill_my_max = (max_rank_opp >= max_rank_my) || has_zha_opp;

        if (!can_kill_opp_max && !can_kill_my_max) {
            std::vector<int> ranks_my, ranks_opp;
            for (int p : combat_my) ranks_my.push_back(st.rank[p]);
            for (int p : combat_opp) ranks_opp.push_back(st.rank[p]);
            std::sort(ranks_my.begin(), ranks_my.end());
            std::sort(ranks_opp.begin(), ranks_opp.end());
            if (ranks_my == ranks_opp) return true;
        }

        bool long_standoff = (st.quiet >= 25 || st.ply >= 140);

        if (!can_kill_opp_max) {
            int opp_dominating = 0;
            for (int p : combat_opp) if (st.rank[p] > max_rank_my) opp_dominating++;
            int my_in_camps = 0;
            for (int p : combat_my) if (is_camp_idx(p)) my_in_camps++;
            bool all_in_camps = (my_in_camps == static_cast<int>(combat_my.size()));
            if (opp_dominating <= 1 &&
                (all_in_camps || (my_in_camps >= 2 && long_standoff))) {
                return true;
            }
        }
        if (!can_kill_my_max) {
            int my_dominating = 0;
            for (int p : combat_my) if (st.rank[p] > max_rank_opp) my_dominating++;
            int opp_in_camps = 0;
            for (int p : combat_opp) if (is_camp_idx(p)) opp_in_camps++;
            bool all_opp_in_camps = (opp_in_camps == static_cast<int>(combat_opp.size()));
            if (my_dominating <= 1 &&
                (all_opp_in_camps || (opp_in_camps >= 2 && long_standoff))) {
                return true;
            }
        }
    }

    // 原型 2: 1v1 追逐死锁
    if (combat_my.size() == 1 && combat_opp.size() == 1 && st.hidden_count == 0 &&
        st.total_dead >= 30) {
        int pos_m = combat_my[0], pos_o = combat_opp[0];
        int r_m = st.rank[pos_m], r_o = st.rank[pos_o];
        if (r_m == r_o) return true;
        int superior_pos, inferior_pos;
        if (r_m > r_o) {
            superior_pos = pos_m;
            inferior_pos = pos_o;
        } else {
            superior_pos = pos_o;
            inferior_pos = pos_m;
        }
        if (is_camp_idx(inferior_pos)) return true;

        std::vector<char> vis(NUM_CELLS, 0);
        std::deque<std::pair<int, int>> q;
        vis[inferior_pos] = 1;
        q.emplace_back(inferior_pos, 0);
        int dist_to_camp = 999;
        while (!q.empty()) {
            auto [cur, d] = q.front();
            q.pop_front();
            if (is_camp_idx(cur)) {
                dist_to_camp = d;
                break;
            }
            const CellNeighbors& nb = nb_of(cur);
            for (int k = 0; k < nb.count; ++k) {
                int nx = nb.neighbors[k];
                if (!vis[nx] && nx != superior_pos) {
                    vis[nx] = 1;
                    q.emplace_back(nx, d + 1);
                }
            }
        }
        if (dist_to_camp <= 2) return true;

        int ir = inferior_pos / COLS, ic = inferior_pos % COLS;
        if ((ir == 0 || ir == 11) && (ic >= 1 && ic <= 3)) return true;
    }

    // 原型 3: 地雷物理阻断
    if (!my_can_flag && !opp_can_flag && !combat_my.empty() && !combat_opp.empty()) {
        bool impassable[NUM_CELLS];
        std::memset(impassable, 0, sizeof(impassable));
        for (int i = 0; i < NUM_CELLS; ++i) {
            if (st.occupied[i] && st.revealed[i] && (st.rank[i] == 21 || st.rank[i] == 22)) {
                impassable[i] = true;
            }
        }
        std::vector<char> vis(NUM_CELLS, 0);
        std::deque<int> q;
        for (int p : combat_my) {
            vis[p] = 1;
            q.push_back(p);
        }
        std::vector<char> opp_pos(NUM_CELLS, 0);
        for (int p : combat_opp) opp_pos[p] = 1;

        bool can_reach_opp = false;
        while (!q.empty()) {
            int cur = q.front();
            q.pop_front();
            if (opp_pos[cur]) {
                can_reach_opp = true;
                break;
            }
            const CellNeighbors& nb = nb_of(cur);
            for (int k = 0; k < nb.count; ++k) {
                int nx = nb.neighbors[k];
                if (!vis[nx] && !impassable[nx]) {
                    vis[nx] = 1;
                    q.push_back(nx);
                }
            }
        }
        if (!can_reach_opp) return true;
    }

    return false;
}

// ---------------------------------------------------------------- evaluate_expert

double eval_expert_cpp(const ExpertState& st, int seat,
                       const ExpertWeights& w, bool ignore_rule_draw) {
    int my = st.seat_color[seat];
    if (my < 0) return 0.0;
    int opp = 1 - my;

    if (is_dead_draw_cpp(st, ignore_rule_draw)) return 0.0;

    // 1. 存活子力统计（COMPOSITION 序）
    int my_cnt[NUM_RANK_KINDS], opp_cnt[NUM_RANK_KINDS];
    for (int k = 0; k < NUM_RANK_KINDS; ++k) {
        my_cnt[k] = RANK_COMP[k] - st.dead_count[my][k];
        opp_cnt[k] = RANK_COMP[k] - st.dead_count[opp][k];
    }

    // 明子划分 + 梯队计数
    std::vector<int> rev_mine, rev_opp;
    bool my_rev_jun = false, my_rev_zha = false, opp_rev_jun = false, opp_rev_zha = false;
    int my_rev_shi = 0, opp_rev_shi = 0;
    for (int i = 0; i < NUM_CELLS; ++i) {
        if (!st.occupied[i] || !st.revealed[i]) continue;
        if (st.color[i] == my) {
            rev_mine.push_back(i);
            if (st.rank[i] == 12) my_rev_jun = true;
            else if (st.rank[i] == 11) my_rev_shi++;
            else if (st.rank[i] == 20) my_rev_zha = true;
        } else {
            rev_opp.push_back(i);
            if (st.rank[i] == 12) opp_rev_jun = true;
            else if (st.rank[i] == 11) opp_rev_shi++;
            else if (st.rank[i] == 20) opp_rev_zha = true;
        }
    }

    // 2. 动态制霸系数与物质估值
    bool opp_has_si = opp_cnt[R_SI] > 0;
    bool my_has_si = my_cnt[R_SI] > 0;
    bool opp_has_jun = opp_cnt[R_JUN] > 0;
    bool my_has_jun = my_cnt[R_JUN] > 0;
    bool opp_has_zha = opp_cnt[R_ZHA] > 0;
    bool my_has_zha = my_cnt[R_ZHA] > 0;
    bool opp_has_gong = opp_cnt[R_GONG] > 0;
    bool my_has_gong = my_cnt[R_GONG] > 0;

    double mpv[32], opv[32];
    for (int i = 0; i < 32; ++i) {
        mpv[i] = w.piece[i];
        opv[i] = w.piece[i];
    }

    if (!opp_has_si && my_has_si) {
        mpv[13] += 20.0;
        if (!opp_has_zha) mpv[13] += 15.0;
    }
    if (!opp_has_si && !opp_has_jun && my_has_jun) mpv[12] += 15.0;
    if (!my_has_si && opp_has_si) {
        opv[13] += 20.0;
        if (!my_has_zha) opv[13] += 15.0;
    }
    if (!my_has_si && !my_has_jun && opp_has_jun) opv[12] += 15.0;

    // 2.2 二线梯队火力网
    double my_echelon = 0.0, opp_echelon = 0.0;
    if (!my_has_si && opp_has_si) {
        double units = (my_rev_jun ? 1.0 : 0.0) + 0.5 * std::min(2, my_rev_shi) +
                       (my_rev_zha ? 0.8 : 0.0);
        my_echelon = std::min(1.0, units / 2.0) * w.echelon_si_compensation;
    }
    if (!opp_has_si && my_has_si) {
        double units = (opp_rev_jun ? 1.0 : 0.0) + 0.5 * std::min(2, opp_rev_shi) +
                       (opp_rev_zha ? 0.8 : 0.0);
        opp_echelon = std::min(1.0, units / 2.0) * w.echelon_si_compensation;
    }

    // 工兵全灭联动
    if (!opp_has_gong && my_has_gong) {
        mpv[21] += 15.0;
        mpv[22] += 25.0;
    } else if (!my_has_gong && opp_has_gong) {
        opv[21] += 15.0;
        opv[22] += 25.0;
    } else if (!my_has_gong && !opp_has_gong) {
        mpv[21] = 10.0;
        opv[21] = 10.0;
        mpv[22] = 10.0;
        opv[22] = 10.0;
    }

    // 炸弹联动定价
    if (w.use_dynamic_bomb) {
        int best = -1;
        for (int k = 0; k < NUM_RANK_KINDS; ++k) {
            if (k == R_LEI || k == R_QI || k == R_ZHA) continue;
            if (opp_cnt[k] <= 0) continue;
            if (best < 0 || opv[RANK_VALS[k]] > opv[RANK_VALS[best]]) best = k;
        }
        if (best >= 0) mpv[20] = opv[RANK_VALS[best]] * w.bomb_ratio;
        else mpv[20] = opv[6] * w.bomb_ratio;  // PAI

        best = -1;
        for (int k = 0; k < NUM_RANK_KINDS; ++k) {
            if (k == R_LEI || k == R_QI || k == R_ZHA) continue;
            if (my_cnt[k] <= 0) continue;
            if (best < 0 || mpv[RANK_VALS[k]] > mpv[RANK_VALS[best]]) best = k;
        }
        if (best >= 0) opv[20] = mpv[RANK_VALS[best]] * w.bomb_ratio;
        else opv[20] = mpv[6] * w.bomb_ratio;
    } else {
        if (opp_has_si || opp_has_jun) mpv[20] += 10.0;
        if (my_has_si || my_has_jun) opv[20] += 10.0;
    }

    double my_material = 0.0, opp_material = 0.0;
    for (int k = 0; k < NUM_RANK_KINDS; ++k) {
        my_material += mpv[RANK_VALS[k]] * my_cnt[k];
    }
    for (int k = 0; k < NUM_RANK_KINDS; ++k) {
        opp_material += opv[RANK_VALS[k]] * opp_cnt[k];
    }
    double score = (my_material - opp_material) + (my_echelon - opp_echelon);

    // 3.1 行营控制与营内围杀
    int my_camps = 0, opp_camps = 0;
    for (int pos : rev_mine) {
        int pr = st.rank[pos];
        if (is_camp_idx(pos)) {
            my_camps++;
            score += w.camp_occ + 5.0;
            if (pos == 17 || pos == 42) score += w.camp_occ * 0.85;  // (3,2) (8,2)
            if (pr == 20) score += 12.0;
            if (pr != 21 && pr != 22) {
                int adj_hidden = 0;
                const CellNeighbors& nb = nb_of(pos);
                for (int k = 0; k < nb.count; ++k) {
                    int np = nb.neighbors[k];
                    if (st.occupied[np] && !st.revealed[np]) adj_hidden++;
                }
                score += (w.camp_occ * 0.06) * adj_hidden;
            }
            int siege = 0;
            const CellNeighbors& nb2 = nb_of(pos);
            for (int k = 0; k < nb2.count; ++k) {
                int np = nb2.neighbors[k];
                if (!st.occupied[np] || !st.revealed[np] || st.color[np] != opp) continue;
                int res = battle_res(pr, st.rank[np]);
                if (res == BR_ATTACKER || res == BR_BOTH) siege++;
            }
            score += w.camp_siege * siege;
        }
        if (is_hq_idx(pos) && pr != 22 && st.hq_locks_pieces) score += w.hq_locked;
    }
    for (int pos : rev_opp) {
        int pr = st.rank[pos];
        if (is_camp_idx(pos)) {
            opp_camps++;
            score -= (w.camp_occ + 5.0);
            if (pos == 17 || pos == 42) score -= w.camp_occ * 0.85;
            if (pr == 20) score -= 12.0;
            if (pr != 21 && pr != 22) {
                int adj_hidden = 0;
                const CellNeighbors& nb = nb_of(pos);
                for (int k = 0; k < nb.count; ++k) {
                    int np = nb.neighbors[k];
                    if (st.occupied[np] && !st.revealed[np]) adj_hidden++;
                }
                score -= (w.camp_occ * 0.06) * adj_hidden;
            }
            int siege = 0;
            const CellNeighbors& nb2 = nb_of(pos);
            for (int k = 0; k < nb2.count; ++k) {
                int np = nb2.neighbors[k];
                if (!st.occupied[np] || !st.revealed[np] || st.color[np] != my) continue;
                int res = battle_res(pr, st.rank[np]);
                if (res == BR_ATTACKER || res == BR_BOTH) siege++;
            }
            score -= w.camp_siege * siege;
        }
        if (is_hq_idx(pos) && pr != 22 && st.hq_locks_pieces) score -= w.hq_locked;
    }

    // 3.1.1 占营比例非线性矩阵增益
    int net_camps = my_camps - opp_camps;
    if (std::abs(net_camps) >= 2) {
        double sign = net_camps > 0 ? 1.0 : -1.0;
        int k = std::min(std::abs(net_camps), 6);
        double camp_factor;
        switch (k) {
            case 2: camp_factor = 1.0; break;
            case 3: camp_factor = 1.5; break;
            case 4: camp_factor = 2.2; break;
            case 5: camp_factor = 2.6; break;
            default: camp_factor = 3.0; break;
        }
        score += sign * camp_factor * w.camp_matrix_weight;
    }

    // 3.2 空行营控制权与中继推进
    double empty_camps_score = 0.0;
    bool used_my[NUM_CELLS], used_opp[NUM_CELLS];
    std::memset(used_my, 0, sizeof(used_my));
    std::memset(used_opp, 0, sizeof(used_opp));

    for (int ci = 0; ci < 10; ++ci) {
        int cp = get_expert_camp_order()[ci];
        if (st.occupied[cp]) continue;

        int my_reach[8], opp_reach[8];
        int my_n = 0, opp_n = 0;
        const CellNeighbors& cnb = nb_of(cp);
        for (int k = 0; k < cnb.count; ++k) {
            int np_ = cnb.neighbors[k];
            if (is_camp_idx(np_)) continue;
            if (!st.occupied[np_] || !st.revealed[np_]) continue;
            if (st.rank[np_] == 21 || st.rank[np_] == 22) continue;
            if (st.color[np_] == my && !used_my[np_]) my_reach[my_n++] = np_;
            else if (st.color[np_] == opp && !used_opp[np_]) opp_reach[opp_n++] = np_;
        }

        // 2 步通畅中继推进（注意：Python 外循环**不 break**，可累积多个候选）
        if (my_n == 0 && opp_n == 0) {
            for (int k = 0; k < cnb.count; ++k) {
                int np_ = cnb.neighbors[k];
                if (st.occupied[np_] || is_camp_idx(np_)) continue;
                const CellNeighbors& nnb = nb_of(np_);
                for (int j = 0; j < nnb.count; ++j) {
                    int n2 = nnb.neighbors[j];
                    if (is_camp_idx(n2)) continue;
                    if (!st.occupied[n2] || !st.revealed[n2]) continue;
                    if (st.rank[n2] == 21 || st.rank[n2] == 22) continue;
                    if (st.color[n2] == my && st.rank[n2] >= 11 && !used_my[n2]) {
                        if (my_n < 8) my_reach[my_n++] = n2;
                        break;
                    } else if (st.color[n2] == opp && st.rank[n2] >= 11 && !used_opp[n2]) {
                        if (opp_n < 8) opp_reach[opp_n++] = n2;
                        break;
                    }
                }
            }
        }

        if (my_n > 0 && opp_n == 0) {
            used_my[my_reach[0]] = true;
            bool has_major = false;
            for (int k = 0; k < my_n; ++k) if (st.rank[my_reach[k]] >= 11) has_major = true;
            empty_camps_score += has_major ? w.camp_occ * 0.25 : w.camp_occ * 0.15;
        } else if (opp_n > 0 && my_n == 0) {
            used_opp[opp_reach[0]] = true;
            bool has_major = false;
            for (int k = 0; k < opp_n; ++k) if (st.rank[opp_reach[k]] >= 11) has_major = true;
            empty_camps_score -= has_major ? w.camp_occ * 0.25 : w.camp_occ * 0.15;
        } else if (my_n > 0 && opp_n > 0) {
            used_my[my_reach[0]] = true;
            used_opp[opp_reach[0]] = true;
            int my_max = 0, opp_max = 0;
            for (int k = 0; k < my_n; ++k) my_max = std::max(my_max, (int)st.rank[my_reach[k]]);
            for (int k = 0; k < opp_n; ++k) opp_max = std::max(opp_max, (int)st.rank[opp_reach[k]]);
            if (my_max > opp_max) empty_camps_score += w.camp_occ * 0.10;
            else if (opp_max > my_max) empty_camps_score -= w.camp_occ * 0.10;
        }
    }
    score += empty_camps_score;

    // 4. 机动力与死子惩罚
    int my_mobility = 0, opp_mobility = 0;
    for (int pos : rev_mine) {
        if (st.rank[pos] == 21 || st.rank[pos] == 22) continue;
        int moves = 0;
        const CellNeighbors& nb = nb_of(pos);
        for (int k = 0; k < nb.count; ++k) {
            int np = nb.neighbors[k];
            if (!st.occupied[np] ||
                (st.revealed[np] && st.color[np] == opp && !is_camp_idx(np))) {
                moves++;
            }
        }
        if (moves == 0 && !is_camp_idx(pos)) {
            score -= (st.rank[pos] >= 11) ? 8.0 : 4.0;
        }
        my_mobility += moves;
        if (is_rail_idx(pos)) my_mobility += 1;
    }
    for (int pos : rev_opp) {
        if (st.rank[pos] == 21 || st.rank[pos] == 22) continue;
        int moves = 0;
        const CellNeighbors& nb = nb_of(pos);
        for (int k = 0; k < nb.count; ++k) {
            int np = nb.neighbors[k];
            if (!st.occupied[np] ||
                (st.revealed[np] && st.color[np] == my && !is_camp_idx(np))) {
                moves++;
            }
        }
        if (moves == 0 && !is_camp_idx(pos)) {
            score += (st.rank[pos] >= 11) ? 8.0 : 4.0;
        }
        opp_mobility += moves;
        if (is_rail_idx(pos)) opp_mobility += 1;
    }
    score += 0.5 * (my_mobility - opp_mobility);

    // 5.1 军旗暴露
    auto can_take_flag = [&](int flag_color, int attacker_rank) -> bool {
        if (attacker_rank == 21) return false;
        if (st.flag_gong_only && attacker_rank != 5) return false;
        if (st.flag_needs_mines_cleared) {
            int mines_left = RANK_COMP[R_LEI] - st.dead_count[flag_color][R_LEI];
            if (mines_left > 0) return false;
        }
        return true;
    };

    int my_flag = -1, opp_flag = -1;
    for (int p : rev_mine) if (st.rank[p] == 22) { my_flag = p; break; }
    for (int p : rev_opp) if (st.rank[p] == 22) { opp_flag = p; break; }

    if (my_flag >= 0) {
        bool threatened = false;
        for (int ap : rev_opp) {
            bool adj = false;
            const CellNeighbors& anb = nb_of(ap);
            for (int k = 0; k < anb.count; ++k) if (anb.neighbors[k] == my_flag) { adj = true; break; }
            if (adj && can_take_flag(my, st.rank[ap])) { threatened = true; break; }
        }
        if (threatened) score -= w.flag_exposed * 1.5;
    }
    if (opp_flag >= 0) {
        bool threatened = false;
        for (int ap : rev_mine) {
            bool adj = false;
            const CellNeighbors& anb = nb_of(ap);
            for (int k = 0; k < anb.count; ++k) if (anb.neighbors[k] == opp_flag) { adj = true; break; }
            if (adj && can_take_flag(opp, st.rank[ap])) { threatened = true; break; }
        }
        if (threatened) score += w.flag_exposed * 1.5;
    }

    // 5.1.1 地雷护旗
    if (w.mine_flag_guard_bonus > 0.0) {
        if (my_flag >= 0) {
            const CellNeighbors& nb = nb_of(my_flag);
            int guards = 0;
            for (int k = 0; k < nb.count; ++k) {
                int np = nb.neighbors[k];
                if (st.occupied[np] && st.revealed[np] && st.color[np] == my && st.rank[np] == 21) guards++;
            }
            // 等价写法：遍历己方明地雷，判断是否在旗的邻域
            score += w.mine_flag_guard_bonus * guards;
        }
        if (opp_flag >= 0) {
            const CellNeighbors& nb = nb_of(opp_flag);
            int guards = 0;
            for (int k = 0; k < nb.count; ++k) {
                int np = nb.neighbors[k];
                if (st.occupied[np] && st.revealed[np] && st.color[np] == opp && st.rank[np] == 21) guards++;
            }
            score -= w.mine_flag_guard_bonus * guards;
        }
    }

    // 5.2 相邻吃子威胁与后手火力护航
    auto has_battery_support = [&](int defender_pos, int defender_color,
                                   int attacker_rank) -> bool {
        const CellNeighbors& nb = nb_of(defender_pos);
        for (int k = 0; k < nb.count; ++k) {
            int np = nb.neighbors[k];
            if (!st.occupied[np] || !st.revealed[np] || st.color[np] != defender_color) continue;
            if (st.rank[np] == 20 || st.rank[np] >= attacker_rank) return true;
        }
        return false;
    };

    for (int pos : rev_opp) {
        int er = st.rank[pos];
        if (er == 22 || er == 21) continue;
        double best_gain = 0.0;
        const CellNeighbors& nb = nb_of(pos);
        for (int k = 0; k < nb.count; ++k) {
            int np = nb.neighbors[k];
            if (is_camp_idx(np)) continue;
            if (!st.occupied[np] || !st.revealed[np] || st.color[np] != my || st.rank[np] == 22) continue;
            int mr = st.rank[np];
            int res = battle_res(er, mr);
            double v = mpv[mr];
            if (has_battery_support(np, my, er)) {
                if (er > mr) continue;
                else if (er == mr) best_gain = std::max(best_gain, v * 0.2);
                else best_gain = std::max(best_gain, v * 0.5);
            } else {
                if (res == BR_ATTACKER) best_gain = std::max(best_gain, v);
                else if (res == BR_BOTH) best_gain = std::max(best_gain, v * 0.5);
            }
        }
        score -= (is_camp_idx(pos) ? w.attack_camp : w.threat) * best_gain;
    }
    for (int pos : rev_mine) {
        int mr = st.rank[pos];
        if (mr == 22 || mr == 21) continue;
        double best_gain = 0.0;
        const CellNeighbors& nb = nb_of(pos);
        for (int k = 0; k < nb.count; ++k) {
            int np = nb.neighbors[k];
            if (is_camp_idx(np)) continue;
            if (!st.occupied[np] || !st.revealed[np] || st.color[np] != opp || st.rank[np] == 22) continue;
            int er = st.rank[np];
            int res = battle_res(mr, er);
            double v = opv[er];
            if (has_battery_support(np, opp, mr)) {
                if (mr > er) continue;
                else if (mr == er) best_gain = std::max(best_gain, v * 0.2);
                else best_gain = std::max(best_gain, v * 0.5);
            } else {
                if (res == BR_ATTACKER) best_gain = std::max(best_gain, v);
                else if (res == BR_BOTH) best_gain = std::max(best_gain, v * 0.5);
            }
        }
        score += (is_camp_idx(pos) ? w.attack_camp : w.attack) * best_gain;
    }

    // 6. 死区势能
    if (w.fortress > 0 && (my_flag >= 0 || opp_flag >= 0)) {
        double fs_my = my_flag >= 0 ? fortress_score_cpp(st, seat) : 0.0;
        double fs_opp = opp_flag >= 0 ? fortress_score_cpp(st, 1 - seat) : 0.0;
        score += w.fortress * (fs_my - fs_opp);
    }

    // 6.1 残局工兵期权与和棋死锁折现
    int total_gong = my_cnt[R_GONG] + opp_cnt[R_GONG];
    if (total_gong <= 2 && st.ply > 60) {
        int my_mines = 0, opp_mines2 = 0;
        for (int i = 0; i < NUM_CELLS; ++i) {
            if (!st.occupied[i] || !st.revealed[i] || st.rank[i] != 21) continue;
            if (st.color[i] == my) my_mines++;
            else opp_mines2++;
        }
        if (my_mines > 0 && opp_mines2 > 0) {
            score *= (total_gong <= 1) ? 0.75 : 0.85;
        }
    }

    // 6.2 无吃子限步时钟衰减
    if (st.no_capture_draw_plies > 0 &&
        st.quiet >= std::max(10, st.no_capture_draw_plies / 2)) {
        double progress = std::min(1.0, static_cast<double>(st.quiet) / st.no_capture_draw_plies);
        score *= std::max(0.05, (1.0 - progress) * (1.0 - progress));
    }

    // 7. 暗子时差
    int my_active = 0, opp_active = 0;
    for (int p : rev_mine) if (st.rank[p] != 21 && st.rank[p] != 22) my_active++;
    for (int p : rev_opp) if (st.rank[p] != 21 && st.rank[p] != 22) opp_active++;
    score += w.hidden_tempo * (my_active - opp_active);

    return score;
}

}  // namespace junqi
