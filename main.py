from mesa.visualization import SpaceRenderer, SolaraViz
from mesa.visualization.components import AgentPortrayalStyle

from src.av_fleet_simulation.app import run
from src.av_fleet_simulation.model import MultiFleetTrafficModel, ModelParameters


def agent_portrayal(agent):
    return AgentPortrayalStyle(color="tab:blue", size=20)

if __name__ == "__main__":
    params = ModelParameters(
            hdv_count=1,
            lanes=[0,2],
            road_length=[0, 1000],
            torus=False
        )
    model = MultiFleetTrafficModel(params)
    
    renderer = SpaceRenderer(model, backend="matplotlib").render(
        agent_portrayal=agent_portrayal,
    )
    
    page = SolaraViz(model, renderer, components=[], model_params={"params": params})

    run(model)