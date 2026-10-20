from mesa.experimental.scenarios import Scenario

from .agent import VehicleParameters


class FleetParameters:
    def __init__(self, n: int, name: str, av_merge_t: float | None = None):
        self.n = n
        self.name = name
        self.av_merge_t = av_merge_t


class TrafficScenario(Scenario):
    rng: int = 42
    lanes: int = 2
    road_length: int = 1000  # cells (m)
    torus: bool = True
    placement: str = "random"  # random, even (equal spacing) or jam (bumper to bumper)
    density: float = 20  # veh/km/lane
    av_share: float = 0.5  # (AV1 + AV2) / all vehicles
    av1_share: float = 0.5  # AV1 / (AV1 + AV2)
    # Dynamic merge sensing: the time gap (s) each AV fleet requires behind it
    # when merging in front of an AV. VehicleParameters.T (1.8 s) is no DMS
    av1_merge_t: float = VehicleParameters.T
    av2_merge_t: float = VehicleParameters.T
    lc_min_gain: float = (
        1.0  # cells/s -> minimin pred. speed gain based on Gipps (1986)
    )
    lc_cooldown: int = 3  # min. steps befgore next lane change
    # s of travel ahead in which a slower leader caps a lane's sustainable speed;
    # calibrated to Knoop et al. (2012), 0.4-0.5 LC/veh-km in free flow
    lc_lookahead: float = 5.0

    @property
    def vehicle_n(self) -> int:
        cells_per_km = 1000 / VehicleParameters.CELL_M
        return round(self.density * self.lanes * self.road_length / cells_per_km)

    @property
    def fleets(self) -> list[FleetParameters]:
        av_n = round(self.vehicle_n * self.av_share)
        av1_n = round(av_n * self.av1_share)
        return [
            FleetParameters(self.vehicle_n - av_n, "HDV"),
            FleetParameters(av1_n, "AV1", self.av1_merge_t),
            FleetParameters(av_n - av1_n, "AV2", self.av2_merge_t),
        ]

    def to_dict(self):
        return self.__dict__


# A scenario for calibrating the basic model with HDV-only car-following on a
# single-lane 1 km ring. Patterns to match:
# - capacity: 2300 pc/h/ln, from c = min(1900 + 20(FFS - 45), 2300)
#   at FFS ~74 mi/h; HCM 6th ed. (TRB 2016), Ch. 12, multilane highways.
#   Tolerance chosen: ± 150.
# - density at capacity (held out): the flow peak should sit at 45 pc/mi/ln
#   ~ 28 veh/km/ln; HCM 6th ed., Ch. 12. Tolerance chosen: ± 3.
# - concave growth (held out): speed std rises along the platoon, concavely;
#   Jiang et al. (2014), Tian et al. (2016).
calibration_hdvs = TrafficScenario(
    rng=42,
    lanes=2,
    road_length=1000,
    torus=True,
    placement="even",
    density=20,
    av_share=0.5,
)
