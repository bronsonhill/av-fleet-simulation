import solara
from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from src.av_fleet_simulation.agent import VehicleParameters
from src.av_fleet_simulation.app import run
from src.av_fleet_simulation.model import (
    MultiFleetTrafficModel,
    TrafficScenario,
)
from src.av_fleet_simulation.scenario import calibration_hdvs


def agent_portrayal(agent):
    return AgentPortrayalStyle(color="tab:blue", size=10)


def make_value_component(*reporters):
    """Show the latest value of each model reporter as a table."""

    def ValueComponent(model):
        df = model.datacollector.get_model_vars_dataframe()
        if df.empty:
            return solara.Markdown("*No data yet, step the model.*")

        latest = df.iloc[-1]
        rows = "\n".join(f"| {r} | {latest[r]:.3f} |" for r in reporters)
        if "flow" in reporters:
            veh_per_h = latest["flow"] * 3600 / VehicleParameters.STEP_S
            rows += f"\n| flow (veh/h/lane) | {veh_per_h:.0f} |"
        if "density" in reporters:
            veh_per_km = latest["density"] * 1000 / VehicleParameters.CELL_M
            rows += f"\n| density (veh/km/lane) | {veh_per_km:.0f} |"
        return solara.Markdown(
            f"**Step {model.steps}**\n\n| Metric | Value |\n|---|---|\n{rows}"
        )

    return ValueComponent


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
    model_df = model.datacollector.get_model_vars_dataframe()
    model_df.to_csv("results/model_data.csv")
