from src.av_fleet_simulation.model import MultiFleetTrafficModel

SIMULATION_STEPS = 2


def run(model: MultiFleetTrafficModel):

    i = 0
    while i < SIMULATION_STEPS:
        print(f"--- step {i} ---")
        model.step()
        i += 1
