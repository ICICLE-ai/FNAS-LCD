"""Architecture search service — wraps the enumeration library."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# ── path setup (mirrors pipeline/auto_nas.py) ──────────────────────────────
_PROJ = Path(__file__).resolve().parents[2]
for _p in (str(_PROJ), str(_PROJ / "model"), str(_PROJ / "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from enumeration.frontier_enum import (  # noqa: E402
    choose_best,
    read_fc_lookup,
    read_lookup,
    search,
)
from scoring.cross_stage_aggregation import aggregate_score  # noqa: E402

# Fixed score combo: mean (inner) x sum (cross-stage).
METRIC = "analytic_ssr_exp_proj_dw_mean"
AGGREGATE = "sum"


def run_search(
    lookup_csv: Path,
    constraint_column: str,
    num_classes: int,
    budget: float,
) -> dict:
    """Run frontier DP search and return the single best architecture.

    Args:
        lookup_csv: Path to the block lookup CSV.
        constraint_column: Column for budget constraint (e.g. 'flops',
                           'jetson_gpu_latency_ms_median').
        num_classes: Number of output classes.
        budget: Constraint ceiling (ms for latency, FLOPs for flops mode).

    Returns:
        Dict with keys: structure_str, score, constraint, flops, params.
    """
    metric, aggregate = METRIC, AGGREGATE
    is_flops = (constraint_column == "flops")
    candidates_per_in_ch = 200
    states_per_key = 3
    max_layers = 14
    fc_resolution = 7

    # Suppress verbose enumeration output during search
    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        grouped = read_lookup(
            str(lookup_csv),
            constraint_column,
            metric,
            candidates_per_in_ch=candidates_per_in_ch,
        )

        # FC block costs
        if not is_flops:
            proj_cost, linear_cost = read_fc_lookup(
                str(lookup_csv), constraint_column, num_classes
            )
            if linear_cost == 0.0 and num_classes != 100:
                _, fallback_linear = read_fc_lookup(
                    str(lookup_csv), constraint_column, 100
                )
                linear_cost = fallback_linear

            def fc_cost_fn(oc: int) -> float:
                return float(proj_cost.get(oc, 0.0)) + float(linear_cost)
        else:
            fc_cost_fn = None  # search() falls back to PlainNet FLOPs

        finals = search(
            grouped=grouped,
            metric=metric,
            agg=aggregate,
            target=budget,
            bin_size=0.1 if not is_flops else 1e6,
            max_factor=1.0,
            states_per_key=states_per_key,
            max_layers=max_layers,
            fc_resolution=fc_resolution,
            skip_stem_in_agg=True,
            fc_cost_fn=fc_cost_fn,
        )
    finally:
        sys.stdout = saved_stdout

    best_list = choose_best(finals, aggregate, budget, tolerance=0.0)
    if not best_list:
        raise RuntimeError(
            f"No architectures found within budget {budget}. "
            "Try increasing the budget."
        )

    state = best_list[0]
    return {
        "structure_str": "".join(state.cfgs),
        "score": round(aggregate_score(state, aggregate), 6),
        "constraint": round(state.constraint, 2),
        "flops": state.flops,
        "params": state.params,
    }
