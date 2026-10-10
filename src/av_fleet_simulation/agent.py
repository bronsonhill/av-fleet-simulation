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
# TODO [DMS]: Alpha is provisionally interpreted as minimum accepted merging gap in cells, confirm if still the case
#
# TODO [DMS]: Add independent DMS config for AV1 & AV2
#
# TODO [MOBIL]: LC_POLITENESS and LC_SAFE_BRAKE not yet incorporated into LC decision rules.
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
        # LC state
        self.target_lane: int | None = None

        self.lc_cooldown = 0

# ---------- vehicle movement ---------- #
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
        self.d_eff = self.anticipate_safe_travel_dist()

        if self.v * VehicleParameters.T > self.d_eff:
            self.a_state = AState.DEFENSE
        else:
            self.a_state = AState.NORMAL

    def anticipate_safe_travel_dist(self) -> float:
        """
        The max distance that can be safely travelled on the next time step.
        """
        leading, d = self._get_leader()
        v_anti = self.get_leading_anticipated_v(leading)

        return d + max(v_anti - VehicleParameters.G_SAFETY, 0)
# -------------------------------------- #


# ---------- vehicle detection & spacing ---------- #
    def _forward_distance(self, x_from: float, x_to: float) -> float:
        """return forward distance form x_from to x_to 
        on toroidal road distances wrap around road boundary"""
        if self.model.scenario.torus:
            road_length = self.space.x_max - self.space.x_min
            return (x_to - x_from) % road_length
        return x_to - x_from

    def _overlaps_vehicle(self, other: Self) -> bool:
        """ retiurn True if self & other physically overlap at currernt long. positions
        lane membership checked seperately"""
        if self.model.scenario.torus:
            road_length = self.space.x_max - self.space.x_min
            self_to_other = self._forward_distance(self.position[0], other.position[0])
            other_to_self = self._forward_distance(other.position[0], self.position[0])
            return self_to_other < self.length or other_to_self < other.length
        self_start, self_end = self.occupied_interval()
        other_start, other_end = other.occupied_interval()
        return self_start < other_end and other_start < self_end

    def _target_lane_clear(self, target_lane: int) -> bool:
        """check whether self could occupy target_lane w/out physically overlapping another vehicle
        uses current positions before longitudinal movemnt (checks physical occupancy only)"""
        for vehicle in self.space.agents:
            if vehicle is self:
                continue
            if vehicle.position[1] != target_lane:
                continue
            if self._overlaps_vehicle(vehicle):
                return False
        return True
    
    def _get_leader(self, lane: int | None = None) -> tuple[Self, float]: #FLAG ==> duplicates some work in _update_leaders (in model.py)
# TODO [mainmerge]: main branch uses model.leadersd to cache current lane leaders, preserve that optimisation &
#provide seperate lookup for hypothetical target lanes
        """Gets the vehicle that is leading self in its lane <-(del?) 
        return nearest leader and its bumper gap in given lane"""
        if lane is None:
            lane = int(self.position[1])
        leader = self
        nearest_distance = float("inf")
        leader_gap = float("inf") 
        for vehicle in self.space.agents:
                if vehicle is self or vehicle.position[1] != lane:
                    continue
                distance = self._forward_distance(self.position[0], vehicle.position[0])
                if not self.model.scenario.torus and distance <= 0: #non-toroidal roads only consider vehicles ahead
                    continue 
                if distance < nearest_distance: #nearest front decides leader
                    nearest_distance = distance
                    leader = vehicle
                    leader_gap = distance - vehicle.length
        return leader, leader_gap

    def _get_follower(self, lane: int | None = None) -> tuple[Self | None, float]:
        """return nearest follower and its bumper gap in given lane"""
        if lane is None:
            lane = int(self.position[1])
        follower = None
        nearest_distance = float("inf")
        follower_gap = float("inf")
        for vehicle in self.space.agents:
            if vehicle is self or vehicle.position[1] != lane:
                continue
            distance = self._forward_distance(vehicle.position[0], self.position[0])
            if not self.model.scenario.torus and distance <= 0: #non-toroidal roads only consider vehicles ahead
                continue
            if distance < nearest_distance:
                nearest_distance = distance
                follower = vehicle
                follower_gap = distance - self.length
        return follower, follower_gap

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

    def _get_desired_gap(self) -> float:
        """
        The desired gap behind the leading car according to Treiber,
        Hennecke & Helbing (2000)
        """
        breaking_term = 0  # TODO: implement breaking term
        return VehicleParameters.G_SAFETY + max(
            0, self.v * VehicleParameters.T + breaking_term
        )

    def occupied_interval(self):
        """Return the longitudinal range occupied by this vehicle."""
        start = self.position[0] - self.length
        end = self.position[0]
        return start, end

    def _required_rear_merge_gap(self, follower):
        """min. acceptable rear gap for LC 
        TODO[dms] when enabled adjust behavioural gap according to follower flet"""
        closing_speed = max(follower.v - self.v, 0)
        safety_gap = VehicleParameters.G_SAFETY + closing_speed * VehicleParameters.T
        behavioural_gap = self.params.alpha
        return max(safety_gap, behavioural_gap)
# ------------------------------------------------- #
    

# ---------- LC predictions ---------- #
    def _valid_adjacent_lanes(self) -> list[float]:
            """return lanes immediately adjacent to current lane."""
            current_lane = self.position[1]
            num_lanes = self.model.scenario.lanes
            lanes = [] 
            for offset in (-1, 1):
                candidate = current_lane + offset
                if 1 <= candidate <= num_lanes:
                    lanes.append(candidate)
            return lanes
        
    def _anticipated_safe_travel_dist_in_lane(self, lane: float) -> float:
        """ predict safe travel dist. if vehicle occupied 'lane'"""
        leader, gap = self._get_leader(lane)
        if leader is self:
            return float("inf")
        v_anti = self.get_leading_anticipated_v(leader) #FLAG ==> predicted speed of targetr lane leader depends on lead'ers existing lane; decide if Vs evaluate prospective lanes before or after other Vs change lanes!!!
        return gap + max(v_anti - VehicleParameters.G_SAFETY, 0)

    def _predicted_velocity_in_lane(self, lane: float) -> float:
        """predict deterministic next-step velocity in prospective lane"""
# TODO [MAIN MERGE]: main branch assigns individual vehicle max speeds (self.v_max); replace VehicleParameters.V_MAX 
# when integrating with updated long. model
        d_eff = self._anticipated_safe_travel_dist_in_lane(lane)
        return min(self.v + VehicleParameters.A_MAX, VehicleParameters.V_MAX, d_eff)
# ------------------------------------ #


# ---------- safety ---------- #
# MOBIL inspired safety criteria for lane changes (Kesting et al. 2007) but adapted to NHM
    def _front_gap_safe(self, target_lane: float) -> bool:
        """check if front gap is safe for lane change (avoid collision w/leader""" #PRELIM ==> approximation (REVISIT)
        leader, gap = self._get_leader(target_lane)
        if leader is self:
            return True
        return gap >= VehicleParameters.G_SAFETY #+ self.v <-(del?)

    def _rear_gap_safe(self, target_lane: float) -> bool:
# TODO [DMS]: replace shared accepted-gap criterion w/fleet-dependent minimum merging gap:
# alpha(ego_type, prospective_follower_type)
#
#keep collision prevention chekcs separate form behavioural gap acceptance criterion
        """check if rear gap is safe for lane change (avoid collision w/follower)"""
        follower, gap = self._get_follower(target_lane)
        if follower is None:
            return True
        #closing_speed = max(follower.v - self.v, 0) #relative clsoing speed
        #required_gap = (VehicleParameters.G_SAFETY + closing_speed * VehicleParameters.T)
        required_gap = self._required_rear_merge_gap(follower)
        return gap >= required_gap

    def _lane_change_safe(self, target_lane: float) -> bool:
        """check if proposed lane is safe
        physical occupancy checked serperately from behavioural front/rear gap acceptance criteria"""
        if not self._target_lane_clear(target_lane):
            return False
        # TODO [dms] make rear gap accept. fleet dependent
        # TODO [safety] add predicted next step collision checks
        return self._front_gap_safe(target_lane) and self._rear_gap_safe(target_lane)
# ---------------------------- #


# ---------- LC incentive ---------- #
#TODO : Gipps/MOBIL incentive

    def _lane_change_gain(self, target_lane: float) -> float:
        """Predicted velocity advantage from changing lanes."""

        current_v = self._predicted_velocity_in_lane(self.position[1])
        target_v = self._predicted_velocity_in_lane(target_lane)

        return target_v - current_v

    def _lane_change_desirable(self, target_lane: float) -> bool: #FLAG ==> currently not used
        gain = self._lane_change_gain(target_lane)
        return gain > VehicleParameters.LC_MIN_GAIN
# ---------------------------------- #


# ---------- LC decisions & execution ---------- #
    def _choose_lane_change(self) -> float | None:
        """choose safest & most advantageous adj. lane
        returns target lane or none if no lane change is safe and desirable"""
        best_lane = None
        best_gain = VehicleParameters.LC_MIN_GAIN

        for target_lane in self._valid_adjacent_lanes():
            if not self._lane_change_safe(target_lane): #Gipps(1986): establish feasibility/safety
                continue
            gain = self._lane_change_gain(target_lane) #Gipps (1986) & MOBIL (Kesting et al. 2007): consider advantage
            if gain > best_gain:
                best_gain = gain
                best_lane = target_lane
        return best_lane

    def plan_lane_change(self) -> None:
        if self.lc_cooldown >0:
            self.lc_cooldown -= 1
            self.target_lane = None
            return 
        self.target_lane = self._choose_lane_change()

    def apply_lane_change(self) -> None:
        if self.target_lane is not None:
            self.position[1] = self.target_lane
            self.lc_cooldown = VehicleParameters.LC_COOLDOWN #cooldown starts when LC happens
        self.target_lane = None
# ---------------------------------------------- #