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
    """
    Parameters a vehicle requires for initialisation.
    Static class variables are constants that are not varied per vehicle or fleet.
    """

    # Real-world scale of the lattice; used to convert output for display.
    CELL_M = 1  # m per cell
    STEP_S = 1.0  # s per step

    # Tian et al. (2015), Table 4, in cells and steps.
    LENGTH = 7  # cells
    A_MAX = 1  # cells/s^2
    V_MAX = 33  # cells/s
    G_SAFETY = 4  # cells; must be >= B_DEFENSE or cars can collide
    B_DEFENSE = 2  # cells/s^2, extra slowdown when defensive
    T = 1.6  # s, desired time gap
    T_STOPPED = 8  # s stopped before slow-to-start applies
    # Probability of slowing down after the speed-up and brake. P_DEFENSIVE = 1, so
    # in a defensive state the defensive slowdown deterministically applies;
    # the other two are stochastic. Tian et al. (2015) write these as sums:
    P_NORMAL = 0.1  # pc
    P_DEFENSIVE = 1.0  # pa + pc (capped at 1)
    P_STOPPED = 0.65  # pb + pc

    #LC params --> need calibration!!!!!
    LC_MIN_GAIN = 1.0 #cells/s pred. speed avg
    LC_SAFE_BRAKE = 2.0 #cells/s^2 aacceptable braking for target follower
    LC_COOLDOWN  = 3 # steps befgore next lane change 
    LC_POLITENESS = 0.5 #MOBIL weighting of follower disadvantage (Kesting et al. 2007)

    def __init__(self, alpha: float, fleet_name: str, initial_position):
        self.alpha = alpha
        self.fleet_name = fleet_name
        self.initial_position = initial_position
        #self.target_lane = None

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

        self._deterministic_velocity()

        self._random_slowdown()

        # update time stationary
        self.t_st = self.t_st + 1 if self.v == 0 else 0

        self._position_change()

    def _deterministic_velocity(self):
        self.v = min(
            self.v + VehicleParameters.A_MAX, VehicleParameters.V_MAX, self.d_eff
        )

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

    def _anticipated_safe_travel_dist_in_lane(self, lane: float) -> float:
        """ predict safe travel dist. if vehicle occupied 'lane'"""
        leader, gap = self._get_leader(lane)
        if leader is self:
            return float("inf")
        v_anti = self.get_leading_anticipated_v(leader)
        return gap + max(v_anti - VehicleParameters.G_SAFETY, 0)

    def _predicted_velocity_in_lane(self, lane: float) -> float:
        """predict deterministic next-step velocity in prospective lane"""
        d_eff = self._anticipated_safe_travel_dist_in_lane(lane)
        return min(self.v + VehicleParameters.A_MAX, VehicleParameters.V_MAX, d_eff)

# ---------- safety ---------- #
# MOBIL inspired safety criteria for lane changes (Kesting et al. 2007) but adapted to NHM
    def _front_gap_safe(self, target_lane: float) -> bool:
        """check if front gap is safe for lane change (avoid collision w/leader"""
        leader, gap = self._get_leader(target_lane)
        if leader is self:
            return True
        return gap >= VehicleParameters.G_SAFETY #+ self.v <-(del?)

    def _rear_gap_safe(self, target_lane: float) -> bool:
        """check if rear gap is safe for lane change (avoid collision w/follower)"""
        follower, gap = self._get_follower(target_lane)
        if follower is None:
            return True
        closing_speed = max(follower.v - self.v, 0) #relative clsoing speed
        required_gap = (VehicleParameters.G_SAFETY + closing_speed * VehicleParameters.T)
        return gap >= required_gap

    def _lane_change_safe(self, target_lane: float) -> bool:
        """check if lane change is safe (avoid collision w/leader and follower)"""
        return self._front_gap_safe(target_lane) and self._rear_gap_safe(target_lane)
# ---------------------------- #

#TODO : Gipps/MOBIL incentive

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
        """Gets the vehicle that is leading self in its lane <-(del?)
        return nearest vehicle ahead in specified lane"""
        if lane is None:
            lane = self.position[1]

        res: tuple[Self, float] = (self, float("inf"))
        vehicles: AgentSet[Self] = self.space.agents

        for vehicle in vehicles:
                # in order to be leading must be in the same lane  <-(del?)
                #if vehicle is not self and vehicle.position[1] == self.position[1]:  <-(del?)
                if vehicle is self or vehicle.position[1] != lane:
                    continue
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
        _, leading_gap = leading._get_leader()
        return min(
            leading_gap,
            leading.v + VehicleParameters.A_MAX,
            VehicleParameters.V_MAX,
        )

    def _get_follower(self, lane: float | None = None) -> tuple[Self | None, float]:
        """return nearest vehicle behind self in specified lane (new follower - MOBIL)"""
        if lane is None:
            lane = self.position[1]

        follower = None
        best_gap = float("inf")

        for vehicle in self.space.agents:
            if vehicle is self or vehicle.position[1] != lane:
                continue
            #gap from follower's front to self's reard
            gap= (self.position[0] - self.length - vehicle.position[0])

            if self.model.scenario.torus and gap < 0:
                gap += self.space.x_max

            if 0 <= gap< best_gap:
                follower = vehicle
                best_gap = gap

        return follower, best_gap

    
    def occupied_interval(self):
        """Return the longitudinal range occupied by this vehicle."""
        start = self.position[0] - self.length
        end = self.position[0]
        return start, end
