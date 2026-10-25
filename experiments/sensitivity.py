"""
Local sensitivity analysis of the DMS result for each parameter.

- Reference set: the DMS experiment's (experiments/dms.py), 24 veh/km/lane,
  50% AVs split evenly, calibrated lane-change parameters.
- Parameters: each moved one step below (P-) and above (P+) its reference
  value, everything else fixed. ±5% where the parameter is continuous; one
  unit where it is an integer on the lattice, which is a larger relative step,
  so compare S across parameters with that in mind.
  TrafficScenario fields:
    density       24 -> 23 / 25   ±1 veh/km/lane (±2 vehicles), ±4.2%; vehicle
                                  counts are integers so ±5% is not reachable
    av_share      0.5 -> 0.475 / 0.525, realised as 23 / 25 of 48 vehicles
    av1_share     0.5 -> 0.475 / 0.525, realised as 11 / 13 of 24 AVs
    lc_lookahead  5 -> 4.75 / 5.25 s
    lc_min_gain   1 -> 0.95 / 1.05 cells/s. Speeds are integers, so 1.05
                  accepts the same gains as 1.0 and P+ should reproduce the
                  reference exactly (S+ = 0); 0.95 lowers the threshold by a
                  whole step
    lc_cooldown   3 -> 2 / 4 steps, integer
  VehicleParameters constants (car following, Tian et al. 2015):
    T             1.8 -> 1.71 / 1.89 s. "No DMS" means merge_t = T, so the
                  DMS-off cell follows T
    V_MAX         27 -> 26 / 28 cells/s, integer
    V_MAX_SD      2 -> 1.9 / 2.1 cells/s
    G_SAFETY      4 -> 3 / 5 cells, integer; stays >= B_DEFENSE
    B_DEFENSE     2 -> 1 / 3 cells/s^2, integer; stays <= G_SAFETY
    P_NORMAL      0.1 -> 0.095 / 0.105
    P_STOPPED     0.65 -> 0.6175 / 0.6825
    T_STOPPED     8 -> 7 / 9 s, integer
  A_MAX (1) is left out: the only integer step down is 0, which stops all
  acceleration. LENGTH is left out because it changes what density means.
- Each setting runs at two DMS cells, neither fleet (merge_t = T) and both
  fleets (0 s), with the same seeds, so differences come from the parameter
  change.
- Outputs, per setting:
    mean_v      mean speed with no DMS (km/h)
    breakdown   share of no-DMS runs whose mean speed falls below BREAKDOWN_V
    dms_effect  breakdown share with both fleets on DMS minus without
- Relative sensitivity, both directions:
      S = (O' - O) / (P' - P) * P / O
  with a 95% bootstrap CI, resampling seeds in pairs across settings. An S
  whose CI spans 0 is not distinguishable from seed noise.

Run from the repo root:
    uv run python -m experiments.sensitivity
    uv run python -m experiments.sensitivity --plot-only
Writes results/sensitivity.csv (one row per run), results/sensitivity_s.csv
(one row per parameter, direction and output) and results/sensitivity.png.
"""

import argparse
import csv
import multiprocessing
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.dms import (
    AV1_SHARE,
    AV_SHARE,
    BREAKDOWN_V,
    DENSITY,
    LANES,
)
from src.av_fleet_simulation.agent import VehicleParameters
from src.av_fleet_simulation.model import MultiFleetTrafficModel, TrafficScenario

# class constants on VehicleParameters, set in each worker before the model is built
VEHICLE_PARAMS = (
    "T",
    "V_MAX",
    "V_MAX_SD",
    "G_SAFETY",
    "B_DEFENSE",
    "P_NORMAL",
    "P_STOPPED",
    "T_STOPPED",
)
REFERENCE = {
    "density": DENSITY,
    "av_share": AV_SHARE,
    "av1_share": AV1_SHARE,
    "lc_lookahead": TrafficScenario.lc_lookahead,
    "lc_min_gain": TrafficScenario.lc_min_gain,
    "lc_cooldown": TrafficScenario.lc_cooldown,
    **{name: getattr(VehicleParameters, name) for name in VEHICLE_PARAMS},
}
STEPS = {  # (P-, P+)
    "density": (23, 25),
    "av_share": (0.475, 0.525),
    "av1_share": (0.475, 0.525),
    "lc_lookahead": (4.75, 5.25),
    "lc_min_gain": (0.95, 1.05),
    "lc_cooldown": (2, 4),
    "T": (1.71, 1.89),
    "V_MAX": (26, 28),
    "V_MAX_SD": (1.9, 2.1),
    "G_SAFETY": (3, 5),
    "B_DEFENSE": (1, 3),
    "P_NORMAL": (0.095, 0.105),
    "P_STOPPED": (0.6175, 0.6825),
    "T_STOPPED": (7, 9),
}
DMS_ON_MERGE_T = 0.0  # s; DMS off uses VehicleParameters.T, which T's steps move
OUTPUTS = ("mean_v", "breakdown", "dms_effect")


def settings() -> list[tuple[str, str, float]]:
    """(param, direction, value) for the reference and each one-step move."""
    out = [("reference", "0", 0.0)]
    for param, (minus, plus) in STEPS.items():
        out += [(param, "-", minus), (param, "+", plus)]
    return out


def effective_value(scenario: TrafficScenario, param: str, value: float) -> float:
    """The value the model actually ran with, after rounding to whole vehicles."""
    if param == "density":
        return scenario.vehicle_n / (scenario.lanes * scenario.road_length / 1000)
    if param == "av_share":
        avs = sum(f.n for f in scenario.fleets if f.name != "HDV")
        return avs / scenario.vehicle_n
    if param == "av1_share":
        n = {f.name: f.n for f in scenario.fleets}
        return n["AV1"] / (n["AV1"] + n["AV2"])
    return value


def run_one(job: tuple[str, str, float, str, int, int, int]) -> dict:
    param, direction, value, dms, seed, warmup, measure = job
    params = dict(REFERENCE)
    if param != "reference":
        params[param] = value
    # every job sets every constant, so a worker's previous job leaves no trace
    for name in VEHICLE_PARAMS:
        setattr(VehicleParameters, name, params.pop(name))
    merge_t = DMS_ON_MERGE_T if dms == "on" else VehicleParameters.T
    scenario = TrafficScenario(
        rng=seed,
        lanes=LANES,
        placement="even",
        av1_merge_t=merge_t,
        av2_merge_t=merge_t,
        **params,
    )
    model = MultiFleetTrafficModel(scenario)
    for _ in range(warmup + measure):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().iloc[warmup:]
    km_h = VehicleParameters.CELL_M / VehicleParameters.STEP_S * 3.6
    veh_per_h = 3600 / VehicleParameters.STEP_S
    lane_changes = df["total_lane_changes"]
    return {
        "param": param,
        "direction": direction,
        "value": value,
        "p_effective": effective_value(scenario, param, value),
        "dms": dms,
        "merge_t": merge_t,
        "seed": seed,
        "mean_v": mean(df["mean_v"]) * km_h,
        "flow": mean(df["flow"]) * veh_per_h,
        "lane_changes": int(lane_changes.iloc[-1] - lane_changes.iloc[0]),
    }


def outputs(off: np.ndarray, on: np.ndarray) -> dict[str, np.ndarray]:
    """Outputs from per-seed mean speeds; the last axis is seeds."""
    breakdown_off = (off < BREAKDOWN_V).mean(axis=-1)
    breakdown_on = (on < BREAKDOWN_V).mean(axis=-1)
    return {
        "mean_v": off.mean(axis=-1),
        "breakdown": breakdown_off,
        "dms_effect": breakdown_on - breakdown_off,
    }


def speeds(df: pd.DataFrame, param: str, direction: str, dms: str):
    """Per-seed mean speeds for one setting and DMS cell, ordered by seed."""
    cell = df[
        (df["param"] == param) & (df["direction"] == direction) & (df["dms"] == dms)
    ]
    return cell.sort_values("seed")["mean_v"].to_numpy()


def sensitivity(o_new, o_ref, p_new: float, p_ref: float):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (o_new - o_ref) / (p_new - p_ref) * p_ref / o_ref


def analyse(df: pd.DataFrame, boot: int, out: Path) -> list[dict]:
    df = df.astype({"direction": str})
    ref_off = speeds(df, "reference", "0", "off")
    ref_on = speeds(df, "reference", "0", "on")
    n = len(ref_off)
    ref = outputs(ref_off, ref_on)

    # paired bootstrap: the same seed indices for reference and setting
    idx = np.random.default_rng(0).integers(0, n, size=(boot, n))
    ref_boot = outputs(ref_off[idx], ref_on[idx])

    print(
        f"\nreference ({n} seeds): " + ", ".join(f"{k} {v:.3f}" for k, v in ref.items())
    )
    rows = []
    for param, direction, _ in settings()[1:]:
        off = speeds(df, param, direction, "off")
        on = speeds(df, param, direction, "on")
        p_new = (
            df[df["param"] == param]
            .query("direction == @direction")["p_effective"]
            .iloc[0]
        )
        p_ref = REFERENCE[param]
        new = outputs(off, on)
        new_boot = outputs(off[idx], on[idx])
        for key in OUTPUTS:
            s_boot = sensitivity(new_boot[key], ref_boot[key], p_new, p_ref)
            lo, hi = np.nanpercentile(s_boot, [2.5, 97.5])
            rows.append(
                {
                    "param": param,
                    "direction": direction,
                    "p": p_new,
                    "output": key,
                    "o": new[key],
                    "o_ref": ref[key],
                    "s": sensitivity(new[key], ref[key], p_new, p_ref),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )

    print(f"\n{'param':<14}{'dir':<4}{'P':>8}  " + "".join(f"{k:>26}" for k in OUTPUTS))
    for param, direction, _ in settings()[1:]:
        cells = [r for r in rows if r["param"] == param and r["direction"] == direction]
        line = f"{param:<14}{direction:<4}{cells[0]['p']:>8.3f}  "
        for r in cells:
            line += f"{r['s']:>8.2f} [{r['ci_lo']:>6.2f}, {r['ci_hi']:>6.2f}]"
        print(line)

    s_out = out.with_name(f"{out.stem}_s.csv")
    with s_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {s_out}")
    return rows


def plot(rows: list[dict], out: Path) -> None:
    """Tornado chart per output: S- and S+ per parameter with bootstrap CIs."""
    params = list(STEPS)
    fig, axes = plt.subplots(
        1, len(OUTPUTS), figsize=(5 * len(OUTPUTS), 1.5 + 0.4 * len(params)), sharey=True
    )
    for ax, key in zip(axes, OUTPUTS):
        for offset, direction, colour in (
            (-0.18, "-", "tab:blue"),
            (0.18, "+", "tab:orange"),
        ):
            cells = [
                next(
                    r
                    for r in rows
                    if r["param"] == p
                    and r["direction"] == direction
                    and r["output"] == key
                )
                for p in params
            ]
            s = [r["s"] for r in cells]
            err = [
                [r["s"] - r["ci_lo"] for r in cells],
                [r["ci_hi"] - r["s"] for r in cells],
            ]
            ax.barh(
                [i + offset for i in range(len(params))],
                s,
                height=0.36,
                xerr=err,
                color=colour,
                label=f"S{direction}",
            )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(params)), params)
        ax.set_xlabel("relative sensitivity S")
        ax.set_title(key)
    axes[0].legend(fontsize=8)
    fig.suptitle("Local sensitivity around the DMS reference set (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--measure", type=int, default=1000)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--out", default="results/sensitivity.csv")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="skip the runs and re-analyse the existing --out CSV",
    )
    args = parser.parse_args()
    out = Path(args.out)

    if not args.plot_only:
        jobs = [
            (param, direction, value, dms, seed, args.warmup, args.measure)
            for param, direction, value in settings()
            for dms in ("off", "on")
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

    rows = analyse(pd.read_csv(out), args.boot, out)
    plot(rows, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
