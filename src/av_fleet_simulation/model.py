from collections import defaultdict
from statistics import mean, stdev
from typing import override

import mesa
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.experimental.scenarios import Scenario

# from src.av_fleet_simulation.agent import VehicleAgent, VehicleParameters
from .agent import VehicleAgent, VehicleParameters


class FleetParameters:
    def __init__(self, alpha: float, n: int, name: str):
        self.alpha = alpha
        self.n = n
        self.name = name


class TrafficScenario(Scenario):
    lanes: int = 2
    road_length: int = 1000  # cells (m)
    torus: bool = True
    placement: str = "random"  # random, even (equal spacing) or jam (bumper to bumper)
    density: float = 20  # veh/km/lane
    av_share: float = 0.5  # (AV1 + AV2) / all vehicles
    av1_share: float = 0.5  # AV1 / (AV1 + AV2)
    hdv_alpha: float = 1.0
    av1_alpha: float = 1.0
    av2_alpha: float = 1.0
    lc_min_gain: float = (
        1.0  # cells/s -> minimin pred. speed gain based on Gipps (1986)
    )
    lc_cooldown: int = 3  # steps befgore next lane change

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

        self.leaders: dict[VehicleAgent, tuple[VehicleAgent, float]] = {}

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
            "total_lane_changes": lambda m: sum(
                agent.lane_change_count for agent in m.agents
            ),
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
        if self.scenario.placement not in ("random", "even", "jam"):
            raise ValueError(f"Unknown placement: {self.scenario.placement}")

    def _init_agents(self) -> None:
        self._slots = self._placement_slots()
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

    def _placement_slots(self) -> list[tuple[int, int]]:
        """
        Fixed initial positions for "even" and "jam" placement, shuffled so the
        fleets mix. Empty for "random".
        """
        placement = self.scenario.placement
        if placement == "random":
            return []

        n = sum(fleet.n for fleet in self.scenario.fleets)
        lanes = self.scenario.lanes
        length = VehicleParameters.LENGTH
        slots = []
        for i, lane in enumerate(range(1, lanes + 1)):
            lane_n = n // lanes + (i < n % lanes)
            for j in range(lane_n):
                if placement == "even":
                    x = length + int(j * self.scenario.road_length / lane_n)
                elif placement == "jam":
                    x = length + j * (length + 1)
                else:
                    raise ValueError(f"Invalid placement value: {placement}")
                slots.append((x, lane))
        self.random.shuffle(slots)
        return slots

    def _get_available_position(self):
        """Gets a position in continuous space not yet occupied."""
        if self.scenario.placement != "random":
            # assumes caller has correct slot count
            return self._slots.pop()

        # get a randomly available position
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
        self._update_leaders()
        # LC phasee
        self.agents.do("plan_lane_change")
        self._resolve_lc_conflicts()
        self.agents.do("apply_lane_change")
        # longitudinal movement phase
        self._update_leaders()
        self.agents.do("update_a_state")
        self.agents.do("move")
        self.datacollector.collect(self)

    def _update_leaders(self) -> None:
        """
        Finds each vehicle's leader and the gap to its rear by sorting each lane
        once.
        """
        lanes = defaultdict(list)
        for agent in self.agents:
            lanes[agent.position[1]].append(agent)

        torus = self.scenario.torus
        self.leaders = {}
        for cars in lanes.values():
            cars.sort(key=lambda a: a.position[0])
            for i, car in enumerate(cars):
                if i + 1 < len(cars):
                    leader = cars[i + 1]
                elif torus and len(cars) > 1:
                    leader = cars[0]
                else:
                    self.leaders[car] = (car, float("inf"))
                    continue

                gap = leader.position[0] - leader.length - car.position[0]
                if torus and gap < 0:
                    gap += self.space.x_max
                self.leaders[car] = (leader, gap)

    def _resolve_lc_conflicts(self):
        """resolve competing LC proposales before vehicles execute their lane changes
        conservative implementation; all og positions are reserved, approved targets are
        reserved, proposals that overlap either are rejected
        """
        # TODO [safety] -> extend checks to predicted next step movement
        # dms affects indiv. gap acceptance not conflict-resolution priority
        proposals = [agent for agent in self.agents if agent.target_lane is not None]
        self.random.shuffle(proposals)
        proposals.sort(key=lambda agent: agent.v, reverse=True)
        reservations = [(agent, agent.position[1]) for agent in self.agents]
        for agent in proposals:
            target_lane = agent.target_lane
            conflict = False
            for other, reserved_lane in reservations:
                if other is agent:
                    continue
                if reserved_lane != target_lane:
                    continue
                if agent._overlaps_vehicle(other):
                    conflict = True
                    break
            if conflict:
                agent.target_lane = None
            else:
                reservations.append((agent, target_lane))
