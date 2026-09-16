#pragma once
// 切片 3：完整专家搜索子树（_negamax + Star1 机会节点 + 走法排序）的 C++ 实现。
//
// 对应 Python 真源：junqi/search.py 的 _negamax / _evaluate_chance_flip / _order_actions
//
// **混合边界**（重要）：C++ 只拥有 `_negamax` 与机会节点这棵子树；
// 迭代加深的**根循环留在 Python**。理由：根循环承载了 `degraded`、
// `avoid`、`exact_root_scores`、`root_scores`、`_is_tactical` 等最易错的语义，
// 留在 Python 可以零风险复用既有实现与既有测试。
#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

#include "expert_qsearch.h"

namespace junqi {

constexpr int TT_FLAG_EXACT = 0;
constexpr int TT_FLAG_LOWER_BOUND = 1;
constexpr int TT_FLAG_UPPER_BOUND = 2;

constexpr int MAX_KILLERS = 2;
constexpr int MAX_HISTORY = 100'000;
constexpr int MAX_SEARCH_DEPTH = 64;

// 走法排序上下文（对应 _score_action 的 tt_move / killers / history 参数）
struct OrderCtx {
    bool has_tt_move{false};
    Action tt_move{};
    int ply_depth{0};
    const std::array<std::array<Action, MAX_KILLERS>, MAX_SEARCH_DEPTH>* killers{nullptr};
    const std::array<int, MAX_SEARCH_DEPTH>* killer_n{nullptr};
    const std::array<int, NUM_CELLS * NUM_CELLS>* history{nullptr};
};

class ExpertSearch : public ExpertQSearch {
public:
    bool stopped{false};

    void reset_stats() { stats = ExpertQSearchStats{}; }
    void clear_heuristics();
    void clear_tt();

    // 相对当前时刻设置时限（毫秒）。<=0 表示不限时。
    void set_deadline_ms(double ms);
    bool deadline_active() const { return has_deadline_; }

    // 顶层入口：从 blob 构建局面后跑 negamax / 机会节点。
    // 每次顶层调用都会重置路径重复栈（Python 侧根循环传的是同一个空集合）。
    double negamax_blob(const std::string& blob, int depth, int ply_depth,
                        double alpha, double beta);
    double chance_flip_blob(const std::string& blob, int pos, int depth, int ply_depth,
                            double alpha, double beta);

private:
    struct TTEntry {
        uint64_t key{0};
        int depth{0};
        double score{0.0};
        int flag{TT_FLAG_EXACT};
        bool used{false};
        bool has_move{false};
        Action move{};
    };

    std::vector<TTEntry> tt_{};
    size_t tt_mask_{0};
    bool has_deadline_{false};
    // 以 steady_clock epoch 起的秒数保存（double），避免浮点→整数 duration 的
    // 隐式转换问题（MSVC 下 time_point + duration<double,milli> 不收敛）
    std::chrono::duration<double> deadline_{0};

    std::array<std::array<Action, MAX_KILLERS>, MAX_SEARCH_DEPTH> killers_{};
    std::array<int, MAX_SEARCH_DEPTH> killer_n_{};
    std::array<int, NUM_CELLS * NUM_CELLS> history_{};

    std::vector<uint64_t> path_{};   // 树内重复局面（LIFO，等价于 Python 的 set）

    double negamax_(JunqiBoard& b, int depth, int ply_depth, double alpha, double beta);
    double chance_flip_(JunqiBoard& b, uint8_t pos, int depth, int ply_depth,
                        double alpha, double beta);

    std::vector<Action> order_actions(const JunqiBoard& b, std::vector<Action>& acts,
                                      const OrderCtx& ctx);
    double score_action_ctx(const JunqiBoard& b, const Action& act, const OrderCtx& ctx) const;

    bool in_path(uint64_t key) const;
    void tt_store(uint64_t key, int depth, double score, int flag, const Action& mv);
    bool tt_lookup(uint64_t key, int depth, double alpha, double beta,
                   bool* has_move, Action* mv, double* out_cut) ;

    void update_cutoff_heuristics(const Action& a, int ply_depth, int depth);

    double _killer_history_bonus(const Action& act) const override;
};

}  // namespace junqi
