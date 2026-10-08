"""
Concave-growth test (Jiang et al. 2014; Tian et al. 2016, Sec. 3.3): a platoon
starts in a jam, the front car accelerates to a fixed speed and holds it, and
the standard deviation of each follower's speed over time is recorded. In the
data, the std grows along the platoon with diminishing increments.

Run from the repo root:
    python -m experiments.platoon
Writes results/platoon.csv with one row per (leader speed, seed, car), and
results/platoon.png with speed std against car number.
"""

import argparse
import csv
import multiprocessing
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev, stdev

import matplotlib.pyplot as plt
from scipy import stats

from src.av_fleet_simulation.agent import VehicleParameters
from src.av_fleet_simulation.model import MultiFleetTrafficModel, TrafficScenario
from src.av_fleet_simulation.scenario import platoon_hdvs

LEADER_SPEEDS_KM_H = [7, 15, 30, 40, 50]  # Jiang et al. (2014)


def to_cells_per_step(km_h: float) -> int:
    """Leader speed on the lattice, rounded to whole cells per step."""
    return round(km_h / 3.6 * VehicleParameters.STEP_S / VehicleParameters.CELL_M)


def run_one(job: tuple[float, int, int, int]) -> list[dict]:
    leader_km_h, seed, warmup, measure = job
    params = dict(platoon_hdvs)
    params.pop("rng")
    model = MultiFleetTrafficModel(TrafficScenario(rng=seed, **params))

    # jam placement is shuffled, so the front car is the one furthest along
    leader = max(model.agents, key=lambda a: a.position[0])
    leader.v_fixed = to_cells_per_step(leader_km_h)

    # car 1 is the leader, then followers by distance behind it; with one lane
    # and no overtaking the order stays fixed
    road = model.scenario.road_length
    platoon = sorted(
        model.agents, key=lambda a: (leader.position[0] - a.position[0]) % road
    )

    speeds = [[] for _ in platoon]
    for t in range(warmup + measure):
        model.step()
        if t >= warmup:
            for i, car in enumerate(platoon):
                speeds[i].append(car.v)

    m_per_s = VehicleParameters.CELL_M / VehicleParameters.STEP_S
    return [
        {
            "leader_km_h": leader_km_h,
            "seed": seed,
            "car": n,
            "mean_km_h": mean(v) * m_per_s * 3.6,
            "std_km_h": pstdev(v) * m_per_s * 3.6,
        }
        for n, v in enumerate(speeds, start=1)
    ]


def ci95(values: list[float]) -> float:
    """Half-width of the 95% confidence interval on the mean (t-distribution)."""
    n = len(values)
    if n < 2:
        return 0.0
    return stats.t.ppf(0.975, n - 1) * stdev(values) / sqrt(n)


def plot(rows: list[dict], out: Path) -> None:
    """Speed std against car number, one line per leader speed, mean over seeds
    with 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cars = sorted({r["car"] for r in rows})
    for leader_km_h in LEADER_SPEEDS_KM_H:
        by_car = [
            [
                r["std_km_h"]
                for r in rows
                if r["leader_km_h"] == leader_km_h and r["car"] == n
            ]
            for n in cars
        ]
        ax.errorbar(
            cars,
            [mean(v) for v in by_car],
            yerr=[ci95(v) for v in by_car],
            marker="o",
            markersize=3,
            capsize=2,
            label=f"leader {leader_km_h} km/h "
            f"({to_cells_per_step(leader_km_h)} m/s)",
        )
    # TODO: fit std_n = a * n^b per leader speed; b < 1 means concave growth

    ax.set_xlabel("car number (1 = leader)")
    ax.set_ylabel("std of speed (km/h)")
    ax.set_title("Speed std along the platoon")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(
        f"T = {VehicleParameters.T} s, V_MAX = {VehicleParameters.V_MAX} ± {VehicleParameters.V_MAX_SD} m/s"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--measure", type=int, default=10000)
    parser.add_argument("--out", default="results/platoon.csv")
    args = parser.parse_args()

    jobs = [
        (leader_km_h, seed, args.warmup, args.measure)
        for leader_km_h in LEADER_SPEEDS_KM_H
        for seed in range(args.seeds)
    ]

    with multiprocessing.Pool() as pool:
        rows = [row for run in pool.map(run_one, jobs) for row in run]

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
