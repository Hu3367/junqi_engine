#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "types.h"
#include "constants.h"
#include "config.h"
#include "board.h"
#include "rules.h"
#include "eval_apk.h"
#include "eval_expert.h"
#include "expert_qsearch.h"
#include "expert_search.h"
#include "apk_engine.h"

namespace py = pybind11;
using namespace junqi;

PYBIND11_MODULE(junqi_core, m) {
    m.doc() = "Junqi C++ High Performance Game Engine (junqi_core)";

    // Enums
    py::enum_<Color>(m, "Color")
        .value("RED", Color::RED)
        .value("BLUE", Color::BLUE)
        .value("NONE", Color::NONE)
        .export_values();

    py::enum_<Rank>(m, "Rank")
        .value("EMPTY", Rank::EMPTY)
        .value("GONG", Rank::GONG)
        .value("PAI", Rank::PAI)
        .value("LIAN", Rank::LIAN)
        .value("YING", Rank::YING)
        .value("TUAN", Rank::TUAN)
        .value("LV", Rank::LV)
        .value("SHI", Rank::SHI)
        .value("JUN", Rank::JUN)
        .value("SI", Rank::SI)
        .value("ZHA", Rank::ZHA)
        .value("LEI", Rank::LEI)
        .value("QI", Rank::QI)
        .export_values();

    py::enum_<ActionKind>(m, "ActionKind")
        .value("FLIP", ActionKind::FLIP)
        .value("MOVE", ActionKind::MOVE)
        .export_values();

    // Structs
    py::class_<Piece>(m, "Piece")
        .def(py::init<>())
        .def_readwrite("rank", &Piece::rank)
        .def_readwrite("color", &Piece::color)
        .def_readwrite("revealed", &Piece::revealed)
        .def("is_empty", &Piece::is_empty)
        .def("__repr__", [](const Piece& p) {
            if (p.is_empty()) return std::string("Piece(EMPTY)");
            return std::string("Piece(") + color_to_str(p.color) + ", " +
                   std::to_string(static_cast<int>(p.rank)) + ", " +
                   (p.revealed ? "rev" : "hid") + ")";
        });

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_static("make_flip", &Action::make_flip)
        .def_static("make_move", &Action::make_move)
        .def_readwrite("kind", &Action::kind)
        .def_readwrite("frm", &Action::frm)
        .def_readwrite("to", &Action::to)
        .def_property_readonly("frm_pos", [](const Action& a) {
            return std::make_pair(a.frm / COLS, a.frm % COLS);
        })
        .def_property_readonly("to_pos", [](const Action& a) {
            return std::make_pair(a.to / COLS, a.to % COLS);
        })
        .def("__eq__", &Action::operator==)
        .def("__repr__", [](const Action& a) {
            if (a.kind == ActionKind::FLIP) {
                return "Action(flip, (" + std::to_string(a.frm / COLS) + "," + std::to_string(a.frm % COLS) + "))";
            }
            return "Action(move, (" + std::to_string(a.frm / COLS) + "," + std::to_string(a.frm % COLS) + ")->(" +
                   std::to_string(a.to / COLS) + "," + std::to_string(a.to % COLS) + "))";
        });

    py::class_<RuleConfig>(m, "RuleConfig")
        .def(py::init<>())
        .def_readwrite("flag_needs_mines_cleared", &RuleConfig::flag_needs_mines_cleared)
        .def_readwrite("flag_needs_all_flipped", &RuleConfig::flag_needs_all_flipped)
        .def_readwrite("flag_gong_only", &RuleConfig::flag_gong_only)
        .def_readwrite("allow_suicide_attack", &RuleConfig::allow_suicide_attack)
        .def_readwrite("hq_locks_pieces", &RuleConfig::hq_locks_pieces)
        .def_readwrite("engineer_rail_turns", &RuleConfig::engineer_rail_turns)
        .def_readwrite("engineer_can_fly_over_pieces", &RuleConfig::engineer_can_fly_over_pieces)
        .def_readwrite("no_capture_draw_plies", &RuleConfig::no_capture_draw_plies)
        .def_readwrite("max_plies", &RuleConfig::max_plies)
        .def_readwrite("repetition_draw_count", &RuleConfig::repetition_draw_count);

    py::class_<ApkSearchStats>(m, "ApkSearchStats")
        .def_readonly("nodes", &ApkSearchStats::nodes)
        .def_readonly("qnodes", &ApkSearchStats::qnodes)
        .def_readonly("tt_hits", &ApkSearchStats::tt_hits)
        .def_readonly("depth_reached", &ApkSearchStats::depth_reached)
        .def_readonly("time_spent_ms", &ApkSearchStats::time_spent_ms)
        .def_readonly("root_scores", &ApkSearchStats::root_scores);

    // Board
    py::class_<JunqiBoard>(m, "JunqiBoard")
        .def(py::init<>())
        .def_readwrite("turn", &JunqiBoard::turn)
        .def_readwrite("ply", &JunqiBoard::ply)
        .def_readwrite("quiet", &JunqiBoard::quiet)
        .def_readwrite("first_flip_done", &JunqiBoard::first_flip_done)
        .def_readwrite("winner", &JunqiBoard::winner)
        .def_readwrite("win_reason", &JunqiBoard::win_reason)
        .def_readwrite("cfg", &JunqiBoard::cfg)
        .def("add_dead", [](JunqiBoard& b, Rank rank, Color color, bool revealed) {
            b.dead.push_back(Piece{rank, color, revealed});
        })
        .def("get_piece", [](const JunqiBoard& b, int r, int c) -> Piece {
            return b.cells[pos_to_idx(r, c)];
        })
        .def("set_piece", [](JunqiBoard& b, int r, int c, Rank rank, Color color, bool revealed) {
            b.cells[pos_to_idx(r, c)] = Piece{rank, color, revealed};
        })
        .def("set_seat_color", [](JunqiBoard& b, int seat, Color color) {
            b.seat_color[seat] = color;
        })
        .def("get_seat_color", [](const JunqiBoard& b, int seat) -> Color {
            return b.seat_color[seat];
        })
        .def("my_color", &JunqiBoard::my_color, py::arg("seat") = -1)
        .def("is_terminal", &JunqiBoard::is_terminal)
        .def("legal_actions", &JunqiBoard::legal_actions)
        .def("apply", &JunqiBoard::apply)
        .def("compute_zobrist", &JunqiBoard::compute_zobrist)
        .def("clone", &JunqiBoard::clone);

    // Evaluation functions
    m.def("eval_apk_pure", &eval_apk_pure, py::arg("board"), py::arg("my_color"));
    m.def("eval_apk_flip_root", &eval_apk_flip_root, py::arg("board"), py::arg("flip_act"), py::arg("my_color"));

    // ---------------------------------------------------------------- 专家评估（切片 1）
    py::class_<ExpertWeights>(m, "ExpertWeights")
        .def(py::init<>())
        .def_readwrite("camp_occ", &ExpertWeights::camp_occ)
        .def_readwrite("hq_locked", &ExpertWeights::hq_locked)
        .def_readwrite("flag_exposed", &ExpertWeights::flag_exposed)
        .def_readwrite("threat", &ExpertWeights::threat)
        .def_readwrite("attack", &ExpertWeights::attack)
        .def_readwrite("attack_camp", &ExpertWeights::attack_camp)
        .def_readwrite("camp_siege", &ExpertWeights::camp_siege)
        .def_readwrite("camp_zone", &ExpertWeights::camp_zone)
        .def_readwrite("fortress", &ExpertWeights::fortress)
        .def_readwrite("hidden_tempo", &ExpertWeights::hidden_tempo)
        .def_readwrite("echelon_si_compensation", &ExpertWeights::echelon_si_compensation)
        .def_readwrite("camp_matrix_weight", &ExpertWeights::camp_matrix_weight)
        .def_readwrite("mine_flag_guard_bonus", &ExpertWeights::mine_flag_guard_bonus)
        .def_readwrite("bomb_ratio", &ExpertWeights::bomb_ratio)
        .def_readwrite("use_dynamic_bomb", &ExpertWeights::use_dynamic_bomb)
        .def_readwrite("bomb_suicide_exchange", &ExpertWeights::bomb_suicide_exchange)
        .def_readwrite("camp_outstrike_bias", &ExpertWeights::camp_outstrike_bias)
        .def("set_piece", [](ExpertWeights& w, int rank_value, double v) {
            if (rank_value < 0 || rank_value >= 32) throw std::out_of_range("rank_value");
            w.piece[rank_value] = v;
        })
        .def("get_piece", [](const ExpertWeights& w, int rank_value) {
            if (rank_value < 0 || rank_value >= 32) throw std::out_of_range("rank_value");
            return w.piece[rank_value];
        });

    m.def("eval_expert_cpp",
          [](const std::string& board_bytes, const std::string& dead_bytes,
             int turn, int ply, int quiet, int seat0, int seat1,
             bool flag_needs_mines_cleared, bool flag_gong_only,
             bool hq_locks_pieces, int no_capture_draw_plies,
             int seat, const ExpertWeights& w, bool ignore_rule_draw) {
              ExpertState st = make_expert_state(
                  board_bytes, dead_bytes, turn, ply, quiet, seat0, seat1,
                  flag_needs_mines_cleared, flag_gong_only, hq_locks_pieces,
                  no_capture_draw_plies);
              return eval_expert_cpp(st, seat, w, ignore_rule_draw);
          },
          py::arg("board_bytes"), py::arg("dead_bytes"), py::arg("turn"),
          py::arg("ply"), py::arg("quiet"), py::arg("seat0"), py::arg("seat1"),
          py::arg("flag_needs_mines_cleared"), py::arg("flag_gong_only"),
          py::arg("hq_locks_pieces"), py::arg("no_capture_draw_plies"),
          py::arg("seat"), py::arg("weights"), py::arg("ignore_rule_draw") = false);

    m.def("fortress_score_cpp",
          [](const std::string& board_bytes, const std::string& dead_bytes,
             int turn, int ply, int quiet, int seat0, int seat1,
             bool flag_needs_mines_cleared, bool flag_gong_only,
             bool hq_locks_pieces, int no_capture_draw_plies, int seat) {
              ExpertState st = make_expert_state(
                  board_bytes, dead_bytes, turn, ply, quiet, seat0, seat1,
                  flag_needs_mines_cleared, flag_gong_only, hq_locks_pieces,
                  no_capture_draw_plies);
              return fortress_score_cpp(st, seat);
          });

    m.def("is_dead_draw_cpp",
          [](const std::string& board_bytes, const std::string& dead_bytes,
             int turn, int ply, int quiet, int seat0, int seat1,
             bool flag_needs_mines_cleared, bool flag_gong_only,
             bool hq_locks_pieces, int no_capture_draw_plies,
             bool ignore_quiet_limit) {
              ExpertState st = make_expert_state(
                  board_bytes, dead_bytes, turn, ply, quiet, seat0, seat1,
                  flag_needs_mines_cleared, flag_gong_only, hq_locks_pieces,
                  no_capture_draw_plies);
              return is_dead_draw_cpp(st, ignore_quiet_limit);
          });

    // ---------------------------------------------------------------- 切片 2：QSearch
    py::class_<ExpertQSearch>(m, "ExpertQSearch")
        .def(py::init<>())
        .def_readwrite("weights", &ExpertQSearch::w)
        .def_readwrite("qsearch_depth", &ExpertQSearch::qsearch_depth)
        .def("qsearch", &ExpertQSearch::qsearch,
             py::arg("board"), py::arg("alpha"), py::arg("beta"), py::arg("depth_left"))
        .def("qsearch_blob", &ExpertQSearch::qsearch_blob,
             py::arg("blob"), py::arg("alpha"), py::arg("beta"), py::arg("depth_left"))
        .def("reset_stats", [](ExpertQSearch& e) { e.stats = ExpertQSearchStats{}; })
        .def_property_readonly("qnodes", [](const ExpertQSearch& e) { return e.stats.qnodes; });

    // ---------------------------------------------------------------- 切片 3：完整搜索子树
    py::class_<ExpertSearch, ExpertQSearch>(m, "ExpertSearch")
        .def(py::init<>())
        .def_readwrite("stopped", &ExpertSearch::stopped)
        .def("reset_stats", &ExpertSearch::reset_stats)
        .def("clear_heuristics", &ExpertSearch::clear_heuristics)
        .def("clear_tt", &ExpertSearch::clear_tt)
        .def("set_deadline_ms", &ExpertSearch::set_deadline_ms, py::arg("ms"))
        .def("negamax_blob", &ExpertSearch::negamax_blob, py::arg("blob"),
             py::arg("depth"), py::arg("ply_depth"), py::arg("alpha"), py::arg("beta"))
        .def("chance_flip_blob", &ExpertSearch::chance_flip_blob, py::arg("blob"),
             py::arg("pos"), py::arg("depth"), py::arg("ply_depth"),
             py::arg("alpha"), py::arg("beta"))
        .def_property_readonly("nodes", [](const ExpertSearch& e) { return e.stats.nodes; })
        .def_property_readonly("qnodes", [](const ExpertSearch& e) { return e.stats.qnodes; })
        .def_property_readonly("chance_nodes", [](const ExpertSearch& e) { return e.stats.chance_nodes; })
        .def_property_readonly("star1_cutoffs", [](const ExpertSearch& e) { return e.stats.star1_cutoffs; })
        .def_property_readonly("pvs_researches", [](const ExpertSearch& e) { return e.stats.pvs_researches; })
        .def_property_readonly("tt_hits", [](const ExpertSearch& e) { return e.stats.tt_hits; });

    // 顺序敏感表的一致性自检入口（供 Python 测试比对，防 C++/Python 漂移）
    m.def("expert_road_neighbors", []() {
        std::vector<std::vector<int>> out;
        const auto& tab = get_expert_road_neighbors();
        for (int i = 0; i < NUM_CELLS; ++i) {
            std::vector<int> row;
            for (int k = 0; k < tab[i].count; ++k) row.push_back(tab[i].neighbors[k]);
            out.push_back(row);
        }
        return out;
    });
    m.def("expert_camp_order", []() {
        const auto& c = get_expert_camp_order();
        return std::vector<int>(c.begin(), c.end());
    });
    m.def("expert_star1_importance", []() {
        // 受限 Star1 展开的战术重要性分档（下标 = RANK_VALS 规范序 SI..QI）
        return std::vector<int>(std::begin(STAR1_IMPORTANCE),
                                std::end(STAR1_IMPORTANCE));
    });
    m.def("expert_zobrist", [](const std::string& blob) {
        // C++ 侧 zobrist 键（供 Python 比对）。
        // 用途：TT 的桶索引是 `key & mask`；若两侧键不同，则**相等性判定一致
        // 但碰撞/淘汰模式不同** ⇒ depth>=3 的子树值会分叉（决策通常不变）。
        JunqiBoard b = board_from_blob(blob);
        return b.compute_zobrist();
    });

    // Search Engine
    py::class_<ApkSearchEngine>(m, "ApkSearchEngine")
        .def(py::init<size_t, std::optional<uint64_t>>(), py::arg("tt_size_power") = 18, py::arg("seed") = std::nullopt)
        .def("clear_tt", &ApkSearchEngine::clear_tt)
        .def("search", &ApkSearchEngine::search,
             py::arg("board"),
             py::arg("level") = "advanced",
             py::arg("depth_override") = -1,
             py::arg("time_limit_ms_override") = -1,
             py::arg("qsearch_depth_override") = -1)
        .def("get_stats", &ApkSearchEngine::get_stats);
}
