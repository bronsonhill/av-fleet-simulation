"""
python3 uv run python calibrate_lc.py
"""
import csv
from itertools import product
from src.av_fleet_simulation.model import MultiFleetTrafficModel, TrafficScenario
from src.av_fleet_simulation.agent import VehicleParameters

GAINS = (0.5, 1.0, 2.0)
COOLDOWNS = (2, 3, 5)
DENSITIES = (20, 40, 60)
SEEDS = (42, 123, 456, 789, 101)
STEPS = 1000
"""GAINS = (1.0,)
COOLDOWNS = (3,)
DENSITIES = (40,)
SEEDS = (42,)
STEPS = 1000"""

def run_simulation(gain, cooldown, density, seed):
    scenario = TrafficScenario(rng=seed, density=density, lc_min_gain=gain, lc_cooldown=cooldown)
    model = MultiFleetTrafficModel(scenario=scenario)
    total_distance = 0
    for _ in range(STEPS):
        model.step()
        total_distance += sum(agent.v for agent in model.agents)
    agents = list(model.agents)
    distance_km = total_distance * VehicleParameters.CELL_M / 1000
    lane_changes = sum(a.lane_change_count for a in agents)
    return {"gain": gain, "cooldown": cooldown, "density": density, "seed": seed,
            "vehicles": len(agents), "lane_changes": sum(a.lane_change_count for a in agents),
            "mean_speed": sum(a.v for a in agents) / len(agents), "lane_changes": lane_changes,
            "distance_km": distance_km, "lc_per_veh_km": (lane_changes / distance_km if distance_km > 0 else 0)}

def main():
    results = []
    for gain, cooldown, density, seed in product(GAINS, COOLDOWNS, DENSITIES, SEEDS):
        result = run_simulation(gain, cooldown, density, seed)
        results.append(result)
        print(f"Gain={gain}, cooldown={cooldown}, "f"density={density}, seed={seed}: "
              f"{result['lane_changes']} lane changes, "f"{result['distance_km']:.2f} vehicle-km, "
              f"{result['lc_per_veh_km']:.4f} LC/vehicle-km")
    with open("lc_calibration_normalised_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
