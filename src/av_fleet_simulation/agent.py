from enum import Enum, auto
from typing import Self

import mesa
from mesa.agentset import AgentSet


class AState(Enum):
    """Acceleration state"""

    NORMAL = auto()
    DEFENSE = auto()
    UNINITIALISED = auto()


class VehicleParameters:
    """Parameters a vehicle requires for initialisation."""

    # Tian et al. (2015), Table 3. One cell = 7.5 m, one step = 1 s.
    LENGTH = 1  # cells
    A_MAX = 1  # cells/s^2
    V_MAX = 5  # cells/s (37.5 m/s, 135 km/h)
    G_SAFETY = 2  # cells; must be >= B_DEFENSE or cars can collide
    B_DEFENSE = 1  # cells/s^2, extra slowdown when defensive
    T = 1.8  # s, desired time gap
    T_STOPPED = 4  # s stopped before slow-to-start applies
    # Probability of slowing down after the speed-up and brake. P_DEFENSIVE = 1, so
    # the defensive slowdown always applies; the other two are random.
    # Tian et al. (2015) write these as sums: pc, pa + pc (capped at 1), pb + pc.
    P_NORMAL = 0.1
    P_DEFENSIVE = 1.0
    P_STOPPED = 0.65

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

        # velocity
        self.v: float = 0
        # effective distance to the leading car on the next timestep
        self.d_eff: float = float("inf")
        # distance to the leading car
        self.d_l: float = float("inf")
        # time steps spent stopped
        self.t_st: int = 0

    def move(self) -> None:
        """Updates vehicle speed and position (Tian et al. 2015, NHM)."""

        # determinstic velocity update
        v = min(self.v + VehicleParameters.A_MAX, VehicleParameters.V_MAX, self.d_eff)

        # random slowdown
        p, slowdown = self._get_p_and_b()
        if self.random.random() < p:
            v = max(v - slowdown, 0)

        # update stationary timer
        self.t_st = self.t_st + 1 if v == 0 else 0
        self.v = v
        self.position[0] += self.v

        if self.space.torus:
            self.position[0] %= self.space.x_max

    def _get_p_and_b(self) -> tuple[float, float]:
        """Slowdown probability and size for the current state."""
        if self.a_state == AState.DEFENSE:
            return (
                VehicleParameters.P_DEFENSIVE,
                VehicleParameters.B_DEFENSE + VehicleParameters.A_MAX,
            )
        elif self.v == 0 and self.t_st > VehicleParameters.T_STOPPED:
            return VehicleParameters.P_STOPPED, VehicleParameters.A_MAX
        elif self.a_state == AState.NORMAL:
            return VehicleParameters.P_NORMAL, VehicleParameters.A_MAX
        else:
            raise ValueError(
                "Acceleration state of vehicle has not been properly initialised"
            )

    def update_a_state(self) -> None:
        """
        Updates the vehicle's acceleration state. Runs for every vehicle before
        any moves, so d_eff is computed from the same snapshot for all of them.
        """
        self.d_eff = self.anticpate_safe_travel_dist()

        if self.v * VehicleParameters.T > self.d_eff:
            self.a_state = AState.DEFENSE
        else:
            self.a_state = AState.NORMAL

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
            if vehicle is not self and vehicle.position[1] == self.position[1]:
                # gap from self's front to the leader's rear
                dist: float = vehicle.position[0] - vehicle.length - self.position[0]

                if self.model.scenario.torus and dist < 0:
                    dist += self.space.x_max

                if dist < res[1]:
                    res = (vehicle, dist)
        return res

    def get_leading_anticipated_v(self, leading: Self) -> float:
        """
        The leader's expected speed next step: it may speed up by A_MAX, but not
        past V_MAX or its own gap (Tian et al. 2015).
        """
        # TODO: implement dynamic velocity anticipation based on vehicle types
        # AVs could use V2V-reported intentions while HDVs use what is below
        if leading is self:
            return 0
        _, leading_gap = leading._get_leading_vehicle()
        return min(
            leading_gap,
            leading.v + VehicleParameters.A_MAX,
            VehicleParameters.V_MAX,
        )

    def occupied_interval(self):
        """Return the longitudinal range occupied by this vehicle."""
        start = self.position[0] - self.length
        end = self.position[0]
        return start, end
