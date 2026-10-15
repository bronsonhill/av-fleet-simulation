"""
Dynamic merge sensing (DMS) grid sweep experiment.

- Independent variables: the merge time gap each AV fleet requires when the
  follower in the target lane is an AV, av1_merge_t x av2_merge_t, swept
  from T = 1.8 s (no DMS: the same gap it needs in front of an
  HDV) down to 0 s (only the standstill buffer G_SAFETY, the physical floor).
- Dependent variables: mean speed per fleet (HDV, AV1, AV2) and system-wide,
  and flow, averaged over the measurement window.
- Fixed:
    - density: at max. calibrated flow, between jam and free flow (HCM density
      at capacity ~28 veh/km/lane).
    - AV share chosen by MPR predictions from literature. 50/50 AV fleet split
      to avoid confounding factors.
    - HDV minimum merge gap: out of scope, could be explored in future work,
      but chose reasoned default inspired by road laws and typical values in
     literature.
- Sensitivity analysis on density: how sensitive are flow and velocity to the
  merge time gap at different densities? ± 2 vehicles/km/lane from the calibrated
  density (--densities).

Run from the repo root:
    uv run python -m experiments.dms
    uv run python -m experiments.dms --densities 26 28 30
Writes results/dms_sweep.csv with one row per (density, av1_merge_t, av2_merge_t,
seed), and one heatmap figure per density.
"""

import argparse
import csv
import multiprocessing
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
from scipy import stats

from src.av_fleet_simulation.agent import VehicleParameters
from src.av_fleet_simulation.model import MultiFleetTrafficModel, TrafficScenario

# s, from no DMS to the physical floor; coarse pass, refine where it bends
MERGE_T_LEVELS = (1.8, 1.35, 0.9, 0.45, 0.0)
DENSITY = 24  # veh/km/lane, peak flow at density from sweep

# TODO: set from the MPR literature value chosen for the report
AV_SHARE = 0.5
AV1_SHARE = 0.5  # equal fleets so neither fleet's size confounds the comparison
LANES = 2

FLEETS = ("hdv", "av1", "av2")
OUTPUTS = ("mean_v", *(f"mean_v_{f}" for f in FLEETS), "flow")

# km/h. Near peak density each run settles in free flow (~85-90 km/h) or breaks
# down into a jam (~60-65 km/h). The threshold sits in the sparse trough of the
# per-run speed histogram.
BREAKDOWN_V = 75
BREAKDOWN_CHECK = (70, 75, 80)


def run_one(job: tuple[float, float, float, int, int, int]) -> dict:
    density, av1_merge_t, av2_merge_t, seed, warmup, measure = job
    scenario = TrafficScenario(
        rng=seed,
        lanes=LANES,
        placement="even",
        density=density,
        av_share=AV_SHARE,
        av1_share=AV1_SHARE,
        av1_merge_t=av1_merge_t,
        av2_merge_t=av2_merge_t,
    )
    model = MultiFleetTrafficModel(scenario)
    for _ in range(warmup + measure):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().iloc[warmup:]
    km_h = VehicleParameters.CELL_M / VehicleParameters.STEP_S * 3.6
    veh_per_h = 3600 / VehicleParameters.STEP_S
    lane_changes = df["total_lane_changes"]
    return {
        "density": density,
        "av1_merge_t": av1_merge_t,
        "av2_merge_t": av2_merge_t,
        "seed": seed,
        "vehicles": len(model.agents),
        **{key: mean(df[key]) * km_h for key in OUTPUTS if key != "flow"},
        "flow": mean(df["flow"]) * veh_per_h,
        "lane_changes": int(lane_changes.iloc[-1] - lane_changes.iloc[0]),
    }


def ci95(values: list[float]) -> float:
    """Half-width of the 95% confidence interval on the mean (t-distribution)."""
    n = len(values)
    if n < 2:
        return 0.0
    return stats.t.ppf(0.975, n - 1) * stdev(values) / sqrt(n)


def cell(
    rows: list[dict], av1_merge_t: float, av2_merge_t: float, key: str
) -> list[float]:
    return [
        r[key]
        for r in rows
        if r["av1_merge_t"] == av1_merge_t and r["av2_merge_t"] == av2_merge_t
    ]


def report(rows: list[dict], density: float) -> None:
    """Print the two tests the research question asks for."""
    lo, hi = MERGE_T_LEVELS[0], MERGE_T_LEVELS[-1]
    print(f"\n--- density {density} veh/km/lane ---")

    # does a fleet's own DMS raise its speed at every rival level?
    for own, rival in (("av1", "av2"), ("av2", "av1")):
        for r_level in MERGE_T_LEVELS:

            def speeds(own, o_level, r_level):
                a1, a2 = (o_level, r_level) if own == "av1" else (r_level, o_level)
                return cell(rows, a1, a2, f"mean_v_{own}")

            off, on = speeds(own, lo, r_level), speeds(own, hi, r_level)
            t = stats.ttest_ind(on, off, equal_var=False)
            print(
                f"{own} merge_t {lo}->{hi} s at {rival}_merge_t={r_level}: "
                f"{mean(off):.2f} -> {mean(on):.2f} km/h (p = {t.pvalue:.3f})"
            )

    # is everyone slower when both fleets merge at the floor?
    for key in ("mean_v", *(f"mean_v_{f}" for f in FLEETS)):
        neither, both = cell(rows, lo, lo, key), cell(rows, hi, hi, key)
        t = stats.ttest_ind(both, neither, equal_var=False)
        print(
            f"{key}: neither {mean(neither):.2f} ± {ci95(neither):.2f}, "
            f"both {mean(both):.2f} ± {ci95(both):.2f} km/h (p = {t.pvalue:.3f})"
        )


def breakdown_share(rows: list[dict], a1: float, a2: float, v: float) -> float:
    """Share of runs in a cell whose mean speed falls below v."""
    speeds = cell(rows, a1, a2, "mean_v")
    return sum(s < v for s in speeds) / len(speeds)


def plot_breakdown(rows: list[dict], density: float, out: Path) -> None:
    """Left: per-run mean speed histogram with the breakdown threshold, which
    justifies the split. Right: breakdown share over the merge-gap grid."""
    fig, (ax_h, ax_p) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_h.hist([r["mean_v"] for r in rows], bins=40, color="tab:blue")
    ax_h.axvline(BREAKDOWN_V, color="black", ls="--", label="threshold")
    ax_h.axvspan(
        min(BREAKDOWN_CHECK),
        max(BREAKDOWN_CHECK),
        color="grey",
        alpha=0.2,
        label="robustness range",
    )
    ax_h.set_xlabel("run mean speed (km/h)")
    ax_h.set_ylabel("runs")
    ax_h.set_title("Per-run speed, all cells")
    ax_h.legend(fontsize=8)

    grid = [
        [breakdown_share(rows, a1, a2, BREAKDOWN_V) for a1 in MERGE_T_LEVELS]
        for a2 in MERGE_T_LEVELS
    ]
    image = ax_p.imshow(grid, origin="lower", cmap="magma_r", vmin=0, vmax=1)
    for i, row in enumerate(grid):
        for j, share in enumerate(row):
            ax_p.text(
                j,
                i,
                f"{share:.2f}",
                ha="center",
                va="center",
                color="white" if share > 0.5 else "black",
                fontsize=8,
            )
    ax_p.set_xticks(range(len(MERGE_T_LEVELS)), MERGE_T_LEVELS)
    ax_p.set_yticks(range(len(MERGE_T_LEVELS)), MERGE_T_LEVELS)
    ax_p.set_xlabel("AV1 merge time gap behind AVs (s)")
    ax_p.set_ylabel("AV2 merge time gap behind AVs (s)")
    seeds = len(cell(rows, MERGE_T_LEVELS[0], MERGE_T_LEVELS[0], "mean_v"))
    ax_p.set_title(f"P(run mean speed < {BREAKDOWN_V} km/h), {seeds} seeds/cell")
    fig.colorbar(image, ax=ax_p, label="breakdown share")

    fig.suptitle(f"Flow breakdown at {density} veh/km/lane")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot(rows: list[dict], density: float, out: Path) -> None:
    """One heatmap of the seed-mean per output over the av1_merge_t x av2_merge_t grid."""
    fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(4 * len(OUTPUTS), 4))
    for ax, key in zip(axes, OUTPUTS):
        grid = [
            [mean(cell(rows, a1, a2, key)) for a1 in MERGE_T_LEVELS]
            for a2 in MERGE_T_LEVELS
        ]
        image = ax.imshow(grid, origin="lower", cmap="viridis")
        ax.set_xticks(range(len(MERGE_T_LEVELS)), MERGE_T_LEVELS)
        ax.set_yticks(range(len(MERGE_T_LEVELS)), MERGE_T_LEVELS)
        ax.set_xlabel("AV1 merge time gap behind AVs (s)")
        ax.set_ylabel("AV2 merge time gap behind AVs (s)")
        ax.set_title(key)
        fig.colorbar(image, ax=ax, label="veh/h/lane" if key == "flow" else "km/h")
    fig.suptitle(f"DMS grid at {density} veh/km/lane")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--densities", type=float, nargs="+", default=[DENSITY])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--measure", type=int, default=1000)
    parser.add_argument("--out", default="results/dms_sweep.csv")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="skip the runs and re-analyse the existing --out CSV",
    )
    args = parser.parse_args()
    out = Path(args.out)

    if args.plot_only:
        with out.open() as f:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        analyse(rows, sorted({r["density"] for r in rows}), out)
        return

    jobs = [
        (density, av1_merge_t, av2_merge_t, seed, args.warmup, args.measure)
        for density in args.densities
        for av1_merge_t in MERGE_T_LEVELS
        for av2_merge_t in MERGE_T_LEVELS
        for seed in range(args.seeds)
    ]

    with multiprocessing.Pool() as pool:
        rows = pool.map(run_one, jobs)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    analyse(rows, args.densities, out)


def analyse(rows: list[dict], densities: list[float], out: Path) -> None:
    for density in densities:
        at_density = [r for r in rows if r["density"] == density]
        report(at_density, density)
        plot(at_density, density, out.with_name(f"{out.stem}_k{density:g}.png"))
        plot_breakdown(
            at_density,
            density,
            out.with_name(f"{out.stem}_k{density:g}_breakdown.png"),
        )


if __name__ == "__main__":
    main()
