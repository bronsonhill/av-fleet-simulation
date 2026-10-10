"""
Fundamental-diagram sweep for calibration: flow and speed against density for
the calibration_hdvs scenario.

Run from the repo root:
    uv run python -m experiments.calibration
Writes results/calibration_sweep.csv with one row per (density, seed), and
results/calibration_sweep.png with flow and speed against density.
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
from src.av_fleet_simulation.scenario import calibration_hdvs

DENSITIES = range(10, 80 + 1)  # veh/km/lane

# HCM 6th ed. (TRB 2016), Ch. 12, basic freeway segments
HCM_DENSITY_AT_CAPACITY = 28  # pc/km/ln (45 pc/mi/ln)
KM_PER_MI = 1.609344


def hcm_capacity(ffs_km_h: float) -> float:
    """HCM freeway capacity in pc/h/ln for a free-flow speed in km/h."""
    ffs = ffs_km_h / KM_PER_MI  # mi/h
    return min(2200 + 10 * (ffs - 50), 2400)


def hcm_breakpoint(ffs_km_h: float) -> float:
    """HCM freeway break-point flow in pc/h/ln (CAF = 1): below it, speed stays
    at FFS; above it, speed falls towards the speed at capacity."""
    ffs = ffs_km_h / KM_PER_MI  # mi/h
    return 1000 + 40 * (75 - ffs)


def hcm_speed_flow(ffs_km_h: float) -> tuple[list[float], list[float]]:
    """
    HCM 6th ed. unified speed-flow curve for a basic freeway segment, from zero
    flow up to capacity, at base conditions (CAF = SAF = 1). Returns flows in
    veh/h/ln and speeds in km/h. The HCM states it for 55 <= FFS <= 75 mi/h.
    """
    ffs = ffs_km_h / KM_PER_MI  # mi/h
    capacity = hcm_capacity(ffs_km_h)  # pc/h/ln
    breakpoint = hcm_breakpoint(ffs_km_h)  # pc/h/ln
    speed_at_capacity = capacity / 45  # mi/h, density at capacity 45 pc/mi/ln

    flows = [capacity * i / 100 for i in range(101)]
    speeds = [
        ffs
        if q <= breakpoint
        else ffs
        - (ffs - speed_at_capacity) * ((q - breakpoint) / (capacity - breakpoint)) ** 2
        for q in flows
    ]
    return flows, [v * KM_PER_MI for v in speeds]


def run_one(job: tuple[float, int, int, int]) -> dict:
    density, seed, warmup, measure = job
    params = {**dict(calibration_hdvs), "density": density}
    params.pop("rng")
    scenario = TrafficScenario(rng=seed, **params)
    model = MultiFleetTrafficModel(scenario)
    for _ in range(warmup + measure):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().iloc[warmup:]
    veh_per_h = 3600 / VehicleParameters.STEP_S
    m_per_s = VehicleParameters.CELL_M / VehicleParameters.STEP_S
    return {
        "density": density,
        "seed": seed,
        "vehicles": len(model.agents),
        "flow_veh_h": mean(df["flow"]) * veh_per_h,
        "speed_km_h": mean(df["mean_v"]) * m_per_s * 3.6,
        "stopped_share": mean(df["stopped_count"]) / len(model.agents),
    }


def ci95(values: list[float]) -> float:
    """Half-width of the 95% confidence interval on the mean (t-distribution)."""
    n = len(values)
    if n < 2:
        return 0.0
    return stats.t.ppf(0.975, n - 1) * stdev(values) / sqrt(n)


def plot(rows: list[dict], out: Path) -> None:
    """Flow and speed against density, and speed against flow with the HCM curve
    overlaid. Density panels: one dot per seed, a line through the
    seed means with 95% confidence intervals, and the HCM targets for comparison.
    Near the flow peak the seeds split between free flow and breakdown, so the
    mean and its interval sit between the two groups; read the dots there.
    """
    # HCM's free-flow speed is the speed at low flow, so take the model's from
    # the lowest density swept.
    lowest = min(r["density"] for r in rows)
    ffs_km_h = mean(r["speed_km_h"] for r in rows if r["density"] == lowest)

    fig, (ax_q, ax_v, ax_vq) = plt.subplots(1, 3, figsize=(16, 4.5))

    densities = sorted({r["density"] for r in rows})
    for ax, key in ((ax_q, "flow_veh_h"), (ax_v, "speed_km_h")):
        ax.scatter(
            [r["density"] for r in rows],
            [r[key] for r in rows],
            color="tab:orange",
            s=10,
            alpha=0.2,
            label="one dot per run",
        )
        by_density = [[r[key] for r in rows if r["density"] == k] for k in densities]
        ax.errorbar(
            densities,
            [mean(v) for v in by_density],
            yerr=[ci95(v) for v in by_density],
            marker="o",
            markersize=4,
            capsize=3,
            label="mean ± 95% CI",
        )
    points = ax_vq.scatter(
        [r["flow_veh_h"] for r in rows],
        [r["speed_km_h"] for r in rows],
        c=[r["density"] for r in rows],
        cmap="viridis",
        s=10,
        alpha=0.7,
        label="one dot per run",
    )
    fig.colorbar(points, ax=ax_vq, label="density (veh/km/lane)")

    hcm_q, hcm_v = hcm_speed_flow(ffs_km_h)
    ax_vq.plot(
        hcm_q, hcm_v, color="grey", ls="--", label=f"HCM, FFS = {ffs_km_h:.0f} km/h"
    )
    ax_vq.axvline(
        hcm_breakpoint(ffs_km_h),
        color="grey",
        ls=":",
        label=f"HCM break-point, {hcm_breakpoint(ffs_km_h):.0f} veh/h",
    )
    ax_vq.set_xlabel("flow (veh/h/lane)")
    ax_vq.set_ylabel("mean speed (km/h)")
    ax_vq.set_title("Speed-flow")
    ax_vq.grid(alpha=0.3)
    ax_vq.legend(fontsize=8)

    ax_q.axhline(
        hcm_capacity(ffs_km_h),
        color="grey",
        ls="--",
        label=f"HCM capacity at FFS = {ffs_km_h:.0f} km/h",
    )
    for ax in (ax_q, ax_v):
        ax.axvline(
            HCM_DENSITY_AT_CAPACITY,
            color="grey",
            ls=":",
            label="HCM density at capacity",
        )
        ax.set_xlabel("density (veh/km/lane)")
        ax.grid(alpha=0.3)
    ax_q.set_ylabel("flow (veh/h/lane)")
    ax_v.set_ylabel("mean speed (km/h)")
    ax_q.set_title("Flow-density")
    ax_v.set_title("Speed-density")
    ax_q.legend(fontsize=8)

    fig.suptitle(
        f"T = {VehicleParameters.T} s, V_MAX = {VehicleParameters.V_MAX} ± {VehicleParameters.V_MAX_SD} m/s"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--measure", type=int, default=1000)
    parser.add_argument("--out", default="results/calibration_sweep.csv")
    args = parser.parse_args()

    jobs = [
        (density, seed, args.warmup, args.measure)
        for density in DENSITIES
        for seed in range(args.seeds)
    ]

    with multiprocessing.Pool() as pool:
        rows = pool.map(run_one, jobs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    plot(rows, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
