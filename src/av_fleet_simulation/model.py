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
    # Fields are annotated class attributes so Scenario registers them in
    # _scenario_defaults, which SolaraViz uses to route model_params here.
    lanes: int = 2
    road_length: int = 200
    torus: bool = True
    hdv_n: int = 5
    hdv_alpha: float = 1.0
    av1_n: int = 5
    av1_alpha: float = 1.0
    av2_n: int = 5
    av2_alpha: float = 1.0

    @property
    def fleets(self) -> list[FleetParameters]:
        return [
            FleetParameters(self.hdv_alpha, self.hdv_n, "HDV"),
            FleetParameters(self.av1_alpha, self.av1_n, "AV1"),
            FleetParameters(self.av2_alpha, self.av2_n, "AV2"),
        ]

    def to_dict(self):
        return self.__dict__


class MultiFleetTrafficModel(mesa.Model):
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
        self.agents.do("update_a_state")

    def _validate_model(self):
        vehicle_count = sum(map(lambda fleet: fleet.n, self.scenario.fleets))

        if (
            vehicle_count * (VehicleParameters.LENGTH + 1)
            > self.space.x_max - self.space.x_min
        ):
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
        self.agents.do("move")
