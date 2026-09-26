from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from src.av_fleet_simulation.app import run
from src.av_fleet_simulation.model import (
    MultiFleetTrafficModel,
    TrafficScenario,
)


def agent_portrayal(agent):
    return AgentPortrayalStyle(color="tab:blue", size=10)


model_params = {
    "rng": 42,
    "lanes": Slider("Lanes", 2, 1, 4),
    "road_length": Slider("Road length", 200, 50, 1000, 10),
    "torus": {"type": "Checkbox", "value": True, "label": "Wrap road"},
    "hdv_n": Slider("HDV count", 15, 0, 30),
    "hdv_alpha": Slider("HDV alpha", 1.0, 0.0, 2.0, 0.1),
    "av1_n": Slider("AV1 count", 15, 0, 30),
    "av1_alpha": Slider("AV1 alpha", 1.0, 0.0, 2.0, 0.1),
    "av2_n": Slider("AV2 count", 15, 0, 30),
    "av2_alpha": Slider("AV2 alpha", 1.0, 0.0, 2.0, 0.1),
}

model = MultiFleetTrafficModel(scenario=TrafficScenario(rng=42))

renderer = SpaceRenderer(model, backend="matplotlib").render(
    agent_portrayal=agent_portrayal,
)

page = SolaraViz(
    model,
    renderer,
    components=[make_plot_component("mean_v"), make_plot_component("mean_d")],
    model_params=model_params,
)


if __name__ == "__main__":
    run(model)
