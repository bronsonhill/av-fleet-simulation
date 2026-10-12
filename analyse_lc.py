from statistics import mean

from src.av_fleet_simulation.model import MultiFleetTrafficModel, TrafficScenario


def analyse_density(density):
    scenario = TrafficScenario(rng=123, density=density, lc_min_gain=2.0, lc_cooldown=3)
    model = MultiFleetTrafficModel(scenario=scenario)

    changes_per_step = []

    for _ in range(1000):
        before = sum(a.lane_change_count for a in model.agents)
        model.step()
        after = sum(a.lane_change_count for a in model.agents)
        changes_per_step.append(after - before)

    agents = list(model.agents)
    counts = [agent.lane_change_count for agent in agents]
    changed = [count for count in counts if count > 0]

    print(f"\nDensity: {density} veh/km/lane")
    print(f"Total vehicles: {len(agents)}")
    print(f"Total lane changes: {sum(counts)}")
    print(f"Vehicles that changed lanes: {len(changed)}")
    print(f"Mean changes per vehicle: {mean(counts):.2f}")
    print(f"Maximum changes by one vehicle: {max(counts)}")
    print("\nLC ACTIVITY")
    print(f"Steps with at least one change: {sum(n > 0 for n in changes_per_step)}")
    print(f"Maximum changes in one step: {max(changes_per_step)}")
    print(f"Mean changes per step: {mean(changes_per_step):.3f}")
    print(f"Changes in first 100 steps: {sum(changes_per_step[:100])}")
    print(f"Changes in last 100 steps: {sum(changes_per_step[-100:])}")

    print("\nTop 10 vehicles by lane changes:")
    for agent in sorted(agents, key=lambda a: a.lane_change_count, reverse=True)[:10]:
        print(f"  {agent.params.fleet_name}: "f"{agent.lane_change_count} changes")

    print("\nLane changes by fleet:")
    for fleet in ("HDV", "AV1", "AV2"):
        fleet_agents = [agent for agent in agents if agent.params.fleet_name == fleet]
        fleet_counts = [agent.lane_change_count for agent in fleet_agents]

        print(f"  {fleet}: "f"{sum(fleet_counts)} total, "f"{sum(c > 0 for c in fleet_counts)}/{len(fleet_agents)} "f"vehicles changed lanes")

for density in (20, 40, 60):
    analyse_density(density)