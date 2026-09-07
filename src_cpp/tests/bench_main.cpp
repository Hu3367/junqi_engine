#include <iostream>
#include <chrono>
#include <cassert>
#include "board.h"
#include "rules.h"
#include "eval_apk.h"
#include "apk_engine.h"

using namespace junqi;

void test_board_and_rules() {
    std::cout << "[Test 1] 验证棋盘与走法生成器..." << std::endl;
    JunqiBoard board;
    board.cells[pos_to_idx(1, 2)] = Piece{Rank::SI, Color::RED, true};
    board.cells[pos_to_idx(1, 3)] = Piece{Rank::JUN, Color::BLUE, true};
    board.seat_color[0] = Color::RED;
    board.seat_color[1] = Color::BLUE;
    board.turn = 0;
    board.first_flip_done = true;

    auto acts = board.legal_actions();
    std::cout << "  合法走法数量: " << acts.size() << std::endl;
    assert(!acts.empty());

    bool found_capture = false;
    for (const auto& a : acts) {
        if (a.kind == ActionKind::MOVE && a.frm == pos_to_idx(1, 2) && a.to == pos_to_idx(1, 3)) {
            found_capture = true;
            break;
        }
    }
    assert(found_capture);
    std::cout << "  [PASS] 成功生成 (1,2) 司令吃 (1,3) 军长动作" << std::endl;

    // 执行吃子
    board.apply(Action::make_move(pos_to_idx(1, 2), pos_to_idx(1, 3)));
    assert(board.cells[pos_to_idx(1, 3)].rank == Rank::SI);
    assert(board.cells[pos_to_idx(1, 2)].is_empty());
    assert(board.dead.size() == 1);
    assert(board.dead[0].rank == Rank::JUN);
    std::cout << "  [PASS] 状态应用与吃子阵亡结算正确" << std::endl;
}

void test_apk_search() {
    std::cout << "[Test 2] 验证 1:1 APK 纯净搜索引擎..." << std::endl;
    JunqiBoard board;
    board.cells[pos_to_idx(1, 2)] = Piece{Rank::SI, Color::RED, true};
    board.cells[pos_to_idx(1, 3)] = Piece{Rank::JUN, Color::BLUE, true};
    board.cells[pos_to_idx(4, 0)] = Piece{Rank::PAI, Color::BLUE, false}; // 暗子
    board.seat_color[0] = Color::RED;
    board.seat_color[1] = Color::BLUE;
    board.turn = 0;
    board.first_flip_done = true;

    ApkSearchEngine engine;
    auto [best_act, score] = engine.search(board, "advanced", 4, 1000, 16);

    std::cout << "  最优动作: kind=" << (best_act.kind == ActionKind::MOVE ? "MOVE" : "FLIP")
              << " (" << (best_act.frm / COLS) << "," << (best_act.frm % COLS) << ")"
              << " -> (" << (best_act.to / COLS) << "," << (best_act.to % COLS) << ")"
              << ", 评分=" << score << std::endl;

    const auto& stats = engine.get_stats();
    std::cout << "  搜索节点: " << stats.nodes << ", QSearch 节点: " << stats.qnodes
              << ", TT 命中: " << stats.tt_hits << ", 耗时: " << stats.time_spent_ms << " ms" << std::endl;

    assert(best_act.kind == ActionKind::MOVE);
    assert(best_act.frm == pos_to_idx(1, 2));
    assert(best_act.to == pos_to_idx(1, 3));
    std::cout << "  [PASS] 搜索引擎成功决策吃大子" << std::endl;
}

void benchmark_move_generation() {
    std::cout << "[Benchmark] 走法生成吞吐量基准测试..." << std::endl;
    JunqiBoard board;
    // 铺设 20 个棋子在中盘
    for (int r = 1; r <= 10; ++r) {
        board.cells[pos_to_idx(r, 0)] = Piece{Rank::PAI, Color::RED, true};
        board.cells[pos_to_idx(r, 4)] = Piece{Rank::PAI, Color::BLUE, true};
    }
    board.seat_color[0] = Color::RED;
    board.seat_color[1] = Color::BLUE;
    board.turn = 0;
    board.first_flip_done = true;

    constexpr int ITERATIONS = 200000;
    auto t0 = std::chrono::steady_clock::now();
    size_t total_acts = 0;

    for (int i = 0; i < ITERATIONS; ++i) {
        auto acts = board.legal_actions();
        total_acts += acts.size();
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_sec = (ITERATIONS / (ms / 1000.0));

    std::cout << "  完成 " << ITERATIONS << " 次走法生成, 耗时 " << ms << " ms ("
              << static_cast<int64_t>(per_sec) << " 次/秒)" << std::endl;
}

int main() {
    std::cout << "=== 军棋 C++ 原生引擎 (junqi_core) 自检与基准测试 ===" << std::endl;
    test_board_and_rules();
    test_apk_search();
    benchmark_move_generation();
    std::cout << "=== 全部自检通过 ===" << std::endl;
    return 0;
}
