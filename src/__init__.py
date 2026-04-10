"""A* TSP：求解器、地图生成、静态图。"""

from .astar_tsp_solver import (
    AStarResult,
    AStarTSPSolver,
    EVENT_DONE,
    EVENT_GOAL,
    EVENT_POP,
    SearchResult,
    run_stepwise_to_end,
)
from .map_generator import (
    TSPGraph,
    build_knn_graph,
    ensure_connected_fallback,
    generate_random_map,
    load_instance,
    random_tsp_instance,
    save_instance,
)
from .feasibility_estimate import (
    FeasibilityEstimate,
    estimate_feasible_closed_tour,
    format_feasibility_log_lines,
)
from .solve_policy import (
    recommend_solver_mode,
    solve_tsp_auto,
    state_space_upper_bound,
)
from .solution_verifier import (
    OptimalityCertificate,
    TourValidation,
    exact_optimal_tour_small_n,
    validate_closed_tour,
)
from .monster_solver import solve_tsp_monster_bnb
from .visualization import draw_tsp, launch_gui, save_map_png

__all__ = [
    "AStarResult",
    "AStarTSPSolver",
    "SearchResult",
    "EVENT_DONE",
    "EVENT_GOAL",
    "EVENT_POP",
    "run_stepwise_to_end",
    "TSPGraph",
    "build_knn_graph",
    "ensure_connected_fallback",
    "generate_random_map",
    "random_tsp_instance",
    "load_instance",
    "save_instance",
    "draw_tsp",
    "launch_gui",
    "save_map_png",
    "recommend_solver_mode",
    "solve_tsp_auto",
    "state_space_upper_bound",
    "TourValidation",
    "OptimalityCertificate",
    "validate_closed_tour",
    "exact_optimal_tour_small_n",
    "solve_tsp_monster_bnb",
    "FeasibilityEstimate",
    "estimate_feasible_closed_tour",
    "format_feasibility_log_lines",
]
