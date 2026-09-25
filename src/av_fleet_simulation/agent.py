from enum import Enum, auto

import mesa


class AState(Enum):
    """Acceleration state"""

    ACCELERATION = auto()
    DECELERATION = auto()
    UNINITIALISED = None


class VehicleParameters:
    """Parameters a vehicle requires for initialisation."""

    LENGTH = 2

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
        self.alpha = self.params.alpha
        self.type = self.params.fleet_name
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

        print("accelerating to position " + str(self.position))

    def update_a_state(self) -> None:
        """updates the vehicles acceleration state"""
        print("update_a_state")

    def occupied_interval(self):
        """Return the longitudinal range occupied by this vehicle."""
        start = self.position[0] - self.length
        end = self.position[0]
        return start, end
