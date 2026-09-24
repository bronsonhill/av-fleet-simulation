from enum import Enum, auto

import mesa


class AState(Enum):
	ACCELERATION = auto()
	DECELERATION = auto()
	UNINITIALISED = None


class VehicleType(Enum):
	HDV = auto()
	AV1 = auto()
	AV2 = auto()


class VehicleParameters:
	"""Parameters a vehicle requires for initialisation."""

	def __init__(self, alpha: float, type: VehicleType, position):
		self.alpha = alpha
		self.type = type
		self.x = position

	def to_dict(self) -> dict:
		return self.__dict__


class VehicleAgent(mesa.Agent):
	"""An agent for all vehicle types."""

	def __init__(self, model: mesa.Model, args) -> None:
		super().__init__(model)

		self.alpha = args.alpha
		self.type = args.type
		self.x = args.x

		self.a_state: AState = AState.UNINITIALISED
		self.a: float = 0
		self.v: float = 0

	def accelerate(self) -> None:
		"""Updates position according to current acceleration and acceleration state."""
		self.x += 1
		print("accelerating to position " + str(self.x))

	def update_a_state(self) -> None:
		"""updates the vehicles acceleration state"""
		print("update_a_state")
