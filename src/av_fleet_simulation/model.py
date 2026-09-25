from typing import override

import mesa
from mesa.experimental.continuous_space import ContinuousSpace

from src.av_fleet_simulation.agent import VehicleAgent, VehicleParameters, VehicleType


class ModelParameters:
    def __init__(self, hdv_count, lanes, road_length, torus):
        self.hdv_count = hdv_count
        self.lanes = lanes
        self.road_length = road_length
        self.torus = torus

    def to_dict(self):
        return self.__dict__


class MultiFleetTrafficModel(mesa.Model):
    """A model of multi-fleet traffic"""

    def __init__(self, params: ModelParameters) -> None:
        super().__init__()
        self.params = params
        self.space = ContinuousSpace(
            dimensions=[[0, params.road_length], [1, params.lanes]],
            random=self.random,
            torus=params.torus,
        )

        self._validate_model()
        self._init_agents()
        self.agents.do("update_a_state")

    def _validate_model(self):
        vehicle_count = self.params.hdv_count
        if (
            vehicle_count * (VehicleParameters.LENGTH + 1)
            > self.space.x_max - self.space.x_min
        ):
            raise ValueError(
                f"Not enough space for vehicles: {vehicle_count} vehicles do \
                            not fit in {self.space.dimensions}"
            )

    def _init_agents(self) -> None:
        self._init_fleet(1, self.params.hdv_count, VehicleType.HDV)
        # self._init_fleet(1, self.params.hdv_count, VehicleType.AV1)
        # self._init_fleet(1, self.params.hdv_count, VehicleType.AV2)

    def _init_fleet(self, alpha: float, n: int, v_type: VehicleType):
        for _ in range(n):
            VehicleAgent.create_agents(
                model=self,
                args=[
                    self.space,
                    VehicleParameters(alpha, v_type, self._get_available_position()),
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
                    f"Randomly assigned position (lane {candidate_lane}, x {candidate_x}) collides with another initialised vehicle - choosing another random initial position"
                )
            else:
                return (candidate_x, candidate_lane)

    @override
    def step(self) -> None:
        """Advance the model by one step"""
        self.agents.do("move")
