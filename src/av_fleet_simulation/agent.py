from enum import Enum, auto
from typing import Self

import mesa


class AState(Enum):
    """Acceleration state"""

    NORMAL = auto()
    DEFENSE = auto()
    UNINITIALISED = auto()


class VehicleParameters:
    """
    Parameters a vehicle requires for initialisation.
    Static class variables are constants that are not varied per vehicle or fleet.
    """

    # Real-world scale of the lattice; used to convert output for display.
    CELL_M = 1  # m per cell
    STEP_S = 1.0  # s per step

    # Tian et al. (2015) Table 4 used as starting point for params.
    LENGTH = 5  # cells
    A_MAX = 1  # cells/s^2
    V_MAX = 27  # cells/s, mean over vehicles
    V_MAX_SD = 2  # cells/s, spread of each vehicle's own max speed
    G_SAFETY = 4  # cells; must be >= B_DEFENSE or cars can collide
    B_DEFENSE = 2  # cells/s^2, extra slowdown when defensive
    T = 1.8  # s, desired time gap
    T_STOPPED = 8  # s stopped before slow-to-start applies
    # Probability of slowing down after the speed-up and brake. P_DEFENSIVE = 1, so
    # in a defensive state the defensive slowdown deterministically applies;
    # the other two are stochastic. Tian et al. (2015) write these as sums:
    P_NORMAL = 0.1  # pc
    P_DEFENSIVE = 1.0  # pa + pc (capped at 1)
    P_STOPPED = 0.65  # pb + pc

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
        # own max speed, an integer as the lattice requires, at least 1 cell/s
        self.v_max: int = max(
            1,
            round(
                self.random.gauss(VehicleParameters.V_MAX, VehicleParameters.V_MAX_SD)
            ),
        )

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

        self._deterministic_velocity()

        self._random_slowdown()

        # update time stationary
        self.t_st = self.t_st + 1 if self.v == 0 else 0

        self._position_change()

    def _deterministic_velocity(self):
        self.v = min(self.v + VehicleParameters.A_MAX, self.v_max, self.d_eff)

    def _random_slowdown(self):
        p, v_slowdown = self._slowdown_probability_and_magnitude()
        if self.random.random() < p:
            self.v = max(self.v - v_slowdown, 0)
        return 0

    def _position_change(self):
        self.position[0] += self.v

        if self.space.torus:
            self.position[0] %= self.space.x_max

    def _slowdown_probability_and_magnitude(self) -> tuple[float, float]:
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
            raise ValueError("Acceleration state of vehicle has not been initialised")

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
        leading, d = self._get_leader()
        v_anti = self.get_leading_anticipated_v(leading)

        return d + max(v_anti - VehicleParameters.G_SAFETY, 0)

    def _get_desired_gap(self) -> float:
        """
        The desired gap behind the leading car according to Treiber,
        Hennecke & Helbing (2000)
        """
        breaking_term = 0  # TODO: implement breaking term
        return VehicleParameters.G_SAFETY + max(
            0, self.v * VehicleParameters.T + breaking_term
        )

    def _get_leader(self) -> tuple[Self, float]:
        """
        Gets the vehicle that is leading self in its lane and the gap to its
        rear. Returns (self, inf) when there is no leader.
        """
        return self.model.leaders[self]

    def get_leading_anticipated_v(self, leading: Self) -> float:
        """
        The leader's expected speed next step: it may speed up by A_MAX, but not
        past its own max speed or its own gap (Tian et al. 2015).
        """
        # TODO: implement dynamic velocity anticipation based on vehicle types
        # AVs could use V2V-reported intentions while HDVs use what is below
        if leading is self:
            return 0
        _, leading_gap = leading._get_leader()
        return min(
            leading_gap,
            leading.v + VehicleParameters.A_MAX,
            leading.v_max,
        )

    def occupied_interval(self):
        """Return the longitudinal range occupied by this vehicle."""
        start = self.position[0] - self.length
        end = self.position[0]
        return start, end
