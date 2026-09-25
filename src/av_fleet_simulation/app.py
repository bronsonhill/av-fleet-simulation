from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle

from src.av_fleet_simulation.model import ModelParameters, MultiFleetTrafficModel

SIMULATION_STEPS = 1


def run(model):

    i = 0
    while i < SIMULATION_STEPS:
        print(f"--- step {i} ---")
        model.step()
        i += 1
