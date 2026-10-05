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
    "road_length": Slider("Road length", 400, 50, 1000, 10),
    "torus": {"type": "Checkbox", "value": True, "label": "Torus"},
    "density": Slider("Density (veh/km/lane)", 20, 5, 80, 1),
    "av_share": Slider("AV share", 0.5, 0.0, 1.0, 0.05),
    "av1_share": Slider("AV1 share of AVs", 0.5, 0.0, 1.0, 0.05),
    "hdv_alpha": Slider("HDV alpha", 1.0, 0.0, 2.0, 0.1),
    "av1_alpha": Slider("AV1 alpha", 1.0, 0.0, 2.0, 0.1),
    "av2_alpha": Slider("AV2 alpha", 1.0, 0.0, 2.0, 0.1),
}

model = MultiFleetTrafficModel(scenario=TrafficScenario(rng=42))

renderer = SpaceRenderer(model, backend="matplotlib").render(
    agent_portrayal=agent_portrayal,
)

page = SolaraViz(
    model,
    renderer,
    components=[
        make_value_component("density", "flow", "mean_v", "stdev_v", "stopped_count"),
        make_plot_component("flow"),
        make_plot_component("mean_v"),
        # make_plot_component("mean_d"),
        make_plot_component("stopped_count"),
        make_plot_component("stdev_v"),
    ],
    model_params=model_params,
)


if __name__ == "__main__":
    run(model)
