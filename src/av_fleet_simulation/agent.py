from enum import Enum, auto
from typing import Self

import mesa
from mesa.agentset import AgentSet


class AState(Enum):
    """Acceleration state"""

    ACCELERATION = auto()
    DECELERATION = auto()
    UNINITIALISED = None


class VehicleParameters:
    """Parameters a vehicle requires for initialisation."""

    LENGTH = 2
    A_MAX = 10
    V_MAX = 50
    G_SAFETY = 2
    T = 1.8

    def __init__(self, alpha: float, fleet_name: str, initial_position):
        self.alpha = alpha
        self.fleet_name = fleet_name
        self.initial_position = initial_position

    def to_dict(self) -> dict:
        return self.__dict__


class VehicleAgent(mesa.experimental.continuous_space.ContinuousSpaceAgent):
    """An agent for all vehicle types."""

    def __init__(self, model: mesa.Model, args) -> None:
        super().__init__(args[0], model)
        self.params = args[1]
        self.position = self.params.initial_position
        self.length = VehicleParameters.LENGTH

        self.a_state: AState = AState.UNINITIALISED
        self.a: float = 0
        self.v: float = 1

    def move(self) -> None:
        """Updates vehicle position."""
        self.v += self.a
        self.position[0] = self.position[0] + 1
        if self.space.torus:
            self.position[0] %= self.space.x_max

    def update_a_state(self) -> None:
        """updates the vehicles acceleration state"""
        print("update_a_state")
        d_anti = self.anticpate_safe_travel_dist()

        if self.v > d_anti / VehicleParameters.T:
            self.a_state = AState.DECELERATION
        else:
            self.a_state = AState.ACCELERATION

        print(self.a_state)

    def anticpate_safe_travel_dist(self) -> float:
        """
        The max distance that can be safely travelled on the next time step.
        """
        leading, d = self._get_leading_vehicle()
        v_anti = self.get_leading_anticipated_v(leading)

        return d + max(v_anti - VehicleParameters.G_SAFETY, 0)

    def _get_leading_vehicle(self) -> tuple[Self, float]:
        """Gets the vehicle that is leading self in it's lane"""

        res: tuple[Self, float] = (self, float("inf"))
        vehicles: AgentSet[Self] = self.space.agents

        for vehicle in vehicles:
            # in order to be leading must be in the same lane
            if vehicle is not self and vehicle.position[1] == vehicle.position[1]:
                dist: float = vehicle.position[0] - self.position[0]

                if self.model.scenario.torus and dist < 0:
                    dist += self.space.x_max

                if dist < res[1]:
                    res = (vehicle, dist)
        return res

    def get_leading_anticipated_v(self, leading: Self) -> float:
        """Gets the velocity anticpated at the next timestep"""
        # There are various ways to calculate v_anti. Knospe et al. introduce a V2V model
        # where the velocity is anticpated based on the leaders leader.
        # below is a simple method that assumes velocity doesn't change.
        # TODO: implement dynamic velocity anticipation based on vehicle types
        return min(leading.v, VehicleParameters.A_MAX)

    def occupied_interval(self):
        """Return the longitudinal range occupied by this vehicle."""
        start = self.position[0] - self.length
        end = self.position[0]
        return start, end
