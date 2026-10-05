from statistics import mean, stdev
from typing import override

import mesa
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.experimental.scenarios import Scenario

from src.av_fleet_simulation.agent import VehicleAgent, VehicleParameters


class FleetParameters:
    def __init__(self, alpha: float, n: int, name: str):
        self.alpha = alpha
        self.n = n
        self.name = name


class TrafficScenario(Scenario):
    lanes: int = 2
    road_length: int = 1000  # cells (m)
    torus: bool = True
    density: float = 20  # veh/km/lane
    av_share: float = 0.5  # (AV1 + AV2) / all vehicles
    av1_share: float = 0.5  # AV1 / (AV1 + AV2)
    hdv_alpha: float = 1.0
    av1_alpha: float = 1.0
    av2_alpha: float = 1.0

    @property
    def vehicle_n(self) -> int:
        cells_per_km = 1000 / VehicleParameters.CELL_M
        return round(self.density * self.lanes * self.road_length / cells_per_km)

    @property
    def fleets(self) -> list[FleetParameters]:
        av_n = round(self.vehicle_n * self.av_share)
        av1_n = round(av_n * self.av1_share)
        return [
            FleetParameters(self.hdv_alpha, self.vehicle_n - av_n, "HDV"),
            FleetParameters(self.av1_alpha, av1_n, "AV1"),
            FleetParameters(self.av2_alpha, av_n - av1_n, "AV2"),
        ]

    def to_dict(self):
        return self.__dict__


class MultiFleetTrafficModel(mesa.Model[VehicleAgent, TrafficScenario]):
    """A model of multi-fleet traffic"""

    def __init__(self, scenario: TrafficScenario) -> None:
        super().__init__(scenario=scenario)
        self.space = ContinuousSpace(
            dimensions=[[0, scenario.road_length], [1, scenario.lanes]],
            random=self.random,
            torus=scenario.torus,
        )

        self._validate_model()
        self._init_agents()
        self._init_datacollector()

    def _init_datacollector(self):
        """Create a `datacollector` and assign it to the model"""
        lane_cells = self.scenario.road_length * self.scenario.lanes

        model_reporters = {
            "flow": lambda m: sum(a.v for a in m.agents) / lane_cells,
            "mean_v": lambda m: mean(agent.v for agent in (a for a in m.agents)),
            # "mean_d": lambda m: mean(agent.d for agent in (a for a in m.agents)),
            "stopped_count": lambda m: sum(
                agent.v == 0 for agent in (a for a in m.agents)
            ),
            "stdev_v": lambda m: stdev(agent.v for agent in (a for a in m.agents)),
            "density": lambda m: len(m.agents) / lane_cells,
        }
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)

    def _validate_model(self):
        """
        Ensures the provided `TrafficScenario` is valid by checking if vehicles
        can fit into the configured road length.
        """
        vehicle_count = sum(fleet.n for fleet in self.scenario.fleets)

        lane_cells = (self.space.x_max - self.space.x_min) * self.scenario.lanes
        if vehicle_count * (VehicleParameters.LENGTH + 1) > lane_cells:
            raise ValueError(
                f"""Not enough space for vehicles: {vehicle_count} vehicles do
                            not fit in {self.space.dimensions}"""
            )

    def _init_agents(self) -> None:
        for fleet_params in self.scenario.fleets:
            self._init_fleet(fleet_params)

    def _init_fleet(self, params: FleetParameters):
        for _ in range(params.n):
            VehicleAgent.create_agents(
                model=self,
                args=[
                    self.space,
                    VehicleParameters(
                        params.alpha, params.name, self._get_available_position()
                    ),
                ],
                n=1,
            )

    def _is_position_occupied(
        self, x_position: float, lane: int, length: float
    ) -> bool:
        """Check whether a vehicle overlaps any existing vehicle in the same lane."""
        occupied_start = x_position - length
        occupied_end = x_position

        for agent in self.agents:
            if agent.position[1] != lane:
                continue

            other_start, other_end = agent.occupied_interval()
            if not (occupied_end < other_start or other_end < occupied_start):
                return True

        return False

    def _get_available_position(self):
        """Gets a position in continuous space not yet occupied."""
        while True:
            candidate_x = self.random.randint(self.space.x_min, self.space.x_max)
            candidate_lane = self.random.randint(self.space.y_min, self.space.y_max)

            if self._is_position_occupied(
                candidate_x,
                candidate_lane,
                VehicleParameters.LENGTH,
            ):
                print(
                    f"Randomly assigned position lane, x: ({candidate_lane}, {candidate_x}) collides with another initialised vehicle - choosing another random initial position"
                )
            else:
                return (candidate_x, candidate_lane)

    @override
    def step(self) -> None:
        """Advance the model by one step"""
        self.agents.do("update_a_state")
        self.agents.do("move")
        self.datacollector.collect(self)
