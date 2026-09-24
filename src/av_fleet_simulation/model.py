from typing import override

import mesa

from src.av_fleet_simulation.vehicle import VehicleAgent, VehicleParameters, VehicleType

class ModelParameters():
    def __init__(self, hdv_count):
        self.hdv_count = hdv_count


class MultiFleetTrafficModel(mesa.Model):
    """A model of multi-fleet traffic"""

    def __init__(self, params: ModelParameters) -> None:
        super().__init__()
        self.num_agents = params.hdv_count
        VehicleAgent.create_agents(
            model=self, 
            args=[VehicleParameters(1, VehicleType.HDV, 0)], 
            n=self.num_agents
            )
        self.agents.do("update_a_state")

    @override
    def step(self) -> None:
        """Advance the model by one step"""
        self.agents.do("accelerate")
