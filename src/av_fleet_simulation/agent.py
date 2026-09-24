from enum import Enum, auto

import mesa


class AState(Enum):
	"""Acceleration state"""
	ACCELERATION = auto()
	DECELERATION = auto()
	UNINITIALISED = None


class VehicleType(Enum):
	HDV = auto()
	AV1 = auto()
	AV2 = auto()


class VehicleParameters:
	"""Parameters a vehicle requires for initialisation."""

	def __init__(self, alpha: float, v_type: VehicleType, position):
		self.alpha = alpha
		self.type = v_type
		self.position = position

	def to_dict(self) -> dict:
		return self.__dict__


class VehicleAgent(mesa.experimental.continuous_space.ContinuousSpaceAgent):
	"""An agent for all vehicle types."""

	def __init__(self, model: mesa.Model, args) -> None:
		super().__init__(args[0], model)
		params = args[1]
		self.alpha = params.alpha
		self.type = params.type
		self.position = params.position

		self.a_state: AState = AState.UNINITIALISED
		self.a: float = 0
		self.v: float = 1

	def move(self) -> None:
		"""Updates vehicle position."""
		self.v += self.a
		self.position[0] += 1
		print("accelerating to position " + str(self.position))

	def update_a_state(self) -> None:
		"""updates the vehicles acceleration state"""
		print("update_a_state")
