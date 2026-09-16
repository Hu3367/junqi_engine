#pragma once
// 切片 2：QSearch（静态搜索）的 C++ 实现。
//
// 对应 Python 真源：junqi/search.py::ExpertSearchEngine._qsearch
// 依赖：legal_actions（rules.cpp）/ battle / evaluate_expert（切片 1）
//
// 与 Python 的两处**有意**差异（都不影响返回值，只影响统计）：
//   1. 不实现 QTT（QSearch 置换表）。C++ 单节点成本 ~µs 级，QTT 的收益
//      已不足以抵掉 zobrist + 查表成本；且它是纯缓存，去掉不改变返回值。
//      代价：stats.qnodes 会比 Python 路径高（不再有 27.5% 的缓存豁免）。
//   2. _score_action 不读 killer/history（它们在搜索过程中动态变化，
//      跨语言同步不划算）。走法顺序只影响剪枝效率，**不影响 minimax 值**。
#include <cstdint>

#include "board.h"
#include "eval_expert.h"

namespace junqi {

constexpr double EXPERT_WIN_SCORE = 1'000'000.0;

struct ExpertQSearchStats {
    uint64_t qnodes{0};
    uint64_t nodes{0};
    uint64_t chance_nodes{0};
    uint64_t star1_cutoffs{0};
    uint64_t pvs_researches{0};
    uint64_t tt_hits{0};
};

class ExpertQSearch {
public:
    ExpertWeights w{};
    int qsearch_depth{16};
    ExpertQSearchStats stats{};

    // 等价于 ExpertSearchEngine._qsearch(state, alpha, beta, depth_left)
    double qsearch(const JunqiBoard& root, double alpha, double beta, int depth_left);

    // 同一入口的**紧凑字节**版本：一次 pybind 调用传入整个局面。
    // 不能用 state_to_cpp（Python 侧 50 次 set_piece ≈ 50 µs，会吃掉全部收益）。
    // 布局见 expert_qsearch.cpp::board_from_blob。
    double qsearch_blob(const std::string& blob, double alpha, double beta, int depth_left);

protected:
    // 走法排序上下文（切片 3 的 ExpertSearch 会设置；切片 2 保持默认=不启用）
    bool cur_has_tt_{false};
    Action cur_tt_{};
    int cur_ply_depth_{0};

    // 钩子：走法排序落到"普通静步"之前，先问派生类有没有杀手/历史加分。
    // 返回 >0 即直接采用；返回 0 表示无，继续用静步分。
    // 只有切片 3 的 ExpertSearch 会覆写（它维护自己的 killer/history 表）。
    virtual double _killer_history_bonus(const Action& act) const { (void)act; return 0.0; }

    double _qsearch(JunqiBoard& b, double alpha, double beta, int depth_left);

    // 等价于 _score_action(act, state, ply_depth, tt_move)。
    // 切片 2 里以 (0, None) 调用且无 killer/history；切片 3 的派生类会传入完整上下文。
    double _score_action(const JunqiBoard& b, const Action& act) const;
};

// 严格对齐 junqi/state.py::GameState.apply 的走子应用（含终局判定）。
// 单独实现而**不复用 JunqiBoard::apply**，因为后者与 Python 有一处语义差异：
// BOTH_DIE 撞军旗时 Python 判攻方胜（winner=turn,"flag"），C++ 未判。
// 改 board.cpp 会影响 APK 引擎，故在此独立实现。
void apply_expert(JunqiBoard& b, const Action& act);

// 等价于 junqi/state.py::GameState._has_any_move
bool has_any_move(const JunqiBoard& b);

// 等价于 junqi/state.py::GameState._flag_attackable([])
bool flag_attackable_no_hidden(const JunqiBoard& b);

// 由 encode_state_blob 的字节流重建 JunqiBoard（切片 2/3 共用的热路径入口）
JunqiBoard board_from_blob(const std::string& blob);

// 由 JunqiBoard 直接求专家估值（内部转换后复用 eval_expert_cpp）
double eval_expert_board(const JunqiBoard& b, int seat, const ExpertWeights& w,
                         bool ignore_rule_draw);

}  // namespace junqi
