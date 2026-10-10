from mesa.experimental.scenarios import Scenario

from .model import VehicleParameters


class FleetParameters:
    def __init__(self, alpha: float, n: int, name: str):
        self.alpha = alpha
        self.n = n
        self.name = name


class TrafficScenario(Scenario):
    rng: int = 42
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


# A scenario for calibrating the basic model with HDV-only car-following on a
# single-lane 1 km ring. Patterns to match:
# - capacity: 2400 pc/h/ln, from c = min(2200 + 10(FFS - 50), 2400)
#   at FFS ~74 mi/h; HCM 6th ed. (TRB 2016), Ch. 12. Tolerance chosen: ± 150.
# - density at capacity (held out): the flow peak should sit at 45 pc/mi/ln
#   ~ 28 veh/km/ln; HCM 6th ed., Ch. 12. Tolerance chosen: ± 3.
# - concave growth (held out): speed std rises along the platoon, concavely;
#   Jiang et al. (2014), Tian et al. (2016).
calibration_hdvs = TrafficScenario(
    rng=42,
    lanes=1,
    road_length=1000,
    torus=True,
    placement="even",
    density=20,
    av_share=0.0,
    hdv_alpha=1.0,
)
