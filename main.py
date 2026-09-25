from mesa.visualization import SpaceRenderer, SolaraViz
from mesa.visualization.components import AgentPortrayalStyle

from src.av_fleet_simulation.app import run
from src.av_fleet_simulation.model import (
    MultiFleetTrafficModel,
    ModelParameters,
    FleetParameters,
)


def agent_portrayal(agent):
    return AgentPortrayalStyle(color="tab:blue", size=10)


if __name__ == "__main__":
    params = ModelParameters(
        [
            FleetParameters(alpha=1, n=5, name="HDV"),
            FleetParameters(alpha=1, n=5, name="AV1"),
            FleetParameters(alpha=1, n=5, name="AV2"),
        ],
        lanes=2,
        road_length=200,
        torus=True,
    )

    model = MultiFleetTrafficModel(params)

    renderer = SpaceRenderer(model, backend="matplotlib").render(
        agent_portrayal=agent_portrayal,
    )

    page = SolaraViz(model, renderer, components=[], model_params={"params": params})

    run(model)
