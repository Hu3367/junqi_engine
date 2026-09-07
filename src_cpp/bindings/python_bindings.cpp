#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "types.h"
#include "constants.h"
#include "config.h"
#include "board.h"
#include "rules.h"
#include "eval_apk.h"
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
