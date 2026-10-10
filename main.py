import solara
from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle

from src.av_fleet_simulation.agent import VehicleParameters
from src.av_fleet_simulation.app import run
from src.av_fleet_simulation.model import MultiFleetTrafficModel
from src.av_fleet_simulation.scenario import TrafficScenario, calibration_hdvs

FLEET_COLORS = {
    "HDV": "tab:gray",
    "AV1": "tab:red",
    "AV2": "tab:green",
}

scenario = calibration_hdvs
# scenario = TrafficScenario()


def agent_portrayal(agent):
    # fallback for fleet names not in the table
    color = FLEET_COLORS.get(agent.params.fleet_name, "tab:blue")
    return AgentPortrayalStyle(color=color, size=10)


def make_table_component(*reporters):
    """Show the latest value of each model reporter as a table."""

    def TableComponent(model):
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

    return TableComponent


model_params = {
    "rng": Slider("Seed", scenario.rng, 0, 100),
    "placement": {
        "type": "Select",
        "label": "Placement",
        "value": scenario.placement,
        "values": ["random", "even", "jam"],
    },
    "lanes": Slider("Lanes", scenario.lanes, 1, 4),
    "road_length": Slider("Road length", scenario.road_length, 50, 1000, 10),
    "torus": {"type": "Checkbox", "value": scenario.torus, "label": "Torus"},
    "density": Slider("Density (veh/km/lane)", scenario.density, 5, 80, 1),
    "av_share": Slider("AV share", scenario.av_share, 0.0, 1.0, 0.05),
    "av1_share": Slider("AV1 share of AVs", scenario.av1_share, 0.0, 1.0, 0.05),
    "hdv_alpha": Slider("HDV alpha", scenario.hdv_alpha, 0.0, 2.0, 0.1),
    "av1_alpha": Slider("AV1 alpha", scenario.av1_alpha, 1.0, 0.0, 2.0, 0.1),
    "av2_alpha": Slider("AV2 alpha", scenario.av2_alpha, 1.0, 0.0, 2.0, 0.1),
}

model = MultiFleetTrafficModel(scenario=scenario)

renderer = SpaceRenderer(model, backend="matplotlib").render(
    agent_portrayal=agent_portrayal,
)

page = SolaraViz(
    model,
    renderer,
    components=[
        make_table_component("density", "flow", "mean_v", "stdev_v", "stopped_count"),
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
