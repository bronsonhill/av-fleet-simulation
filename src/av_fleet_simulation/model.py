from collections import defaultdict
from statistics import mean, stdev
from typing import override

import mesa
from mesa.experimental.continuous_space import ContinuousSpace

from src.av_fleet_simulation.agent import VehicleAgent, VehicleParameters

from .scenario import FleetParameters, TrafficScenario


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
        """
        Create a `datacollector` and assign it to the model. Collects system
        level data.
        """
        lane_cells = self.scenario.road_length * self.scenario.lanes

        def relevant_agents(agents, fleet_name):
            if fleet_name:
                agents = [
                    agent for agent in agents if agent.params.fleet_name == fleet_name
                ]
            return agents

        def flow(fleet_name: str = ""):
            """
            Flow in veh/cell/step, optionally filtered by fleet name.
            Flow: q = k * v, density times mean velocity.
            k = n / lane_cells, number of vehicles divided by total lane cells.
            v = sum of v / n
            So n cancels out, leaving sum of v / lane_cells, which is the total distance.
            """

            def _flow(model: MultiFleetTrafficModel):
                agents = relevant_agents(list(model.agents), fleet_name)
                return sum(a.v for a in agents) / lane_cells

            return _flow

        def mean_v(fleet_name: str = ""):
            def _mean_v(model: MultiFleetTrafficModel):
                agents = relevant_agents(list(model.agents), fleet_name)
                if not agents:
                    return 0
                return mean(agent.v for agent in (a for a in agents))

            return _mean_v

        def stopped_count(fleet_name: str = ""):
            def _stopped_count(model: MultiFleetTrafficModel):
                agents = relevant_agents(list(model.agents), fleet_name)
                return sum(agent.v == 0 for agent in (a for a in agents))

            return _stopped_count

        model_reporters = {
            "flow": flow(),
            "flow_hdv": flow("HDV"),
            "flow_av1": flow("AV1"),
            "flow_av2": flow("AV2"),
            "mean_v": mean_v(),
            "mean_v_hdv": mean_v("HDV"),
            "mean_v_av1": mean_v("AV1"),
            "mean_v_av2": mean_v("AV2"),
            # "mean_d": lambda m: mean(agent.d for agent in (a for a in m.agents)),
            "stopped_count": stopped_count(),
            "stopped_count_hdv": stopped_count("HDV"),
            "stopped_count_av1": stopped_count("AV1"),
            "stopped_count_av2": stopped_count("AV2"),
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
