from typing import override

import mesa
from mesa.experimental.continuous_space import ContinuousSpace

from src.av_fleet_simulation.agent import VehicleAgent, VehicleParameters, VehicleType

class ModelParameters():
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
        self.space = ContinuousSpace([params.road_length, params.lanes], random=self.random, torus=params.torus)
        self._init_agents()
        
        self.agents.do("update_a_state")

    def _init_agents(self) -> None:


        # First, check if there's enough room for all the requested agents in the environment

        # Second, create each fleet
        self._init_fleet(1, self.params.hdv_count, VehicleType.HDV)
        # self._init_fleet(1, self.params.hdv_count, VehicleType.AV1)
        # self._init_fleet(1, self.params.hdv_count, VehicleType.AV2)
        

    def _init_fleet(self, alpha, n, v_type: VehicleType):
        VehicleAgent.create_agents(
            model=self,
            args=[self.space, VehicleParameters(alpha, v_type, self._get_available_position())],
            n=n
        )

    def _get_available_position(self):
        """"""
        return (self.random.randint(self.space.dimensions[0][0], self.space.dimensions[0][1]), 1)
        

    @override
    def step(self) -> None:
        """Advance the model by one step"""
        self.agents.do("move")
