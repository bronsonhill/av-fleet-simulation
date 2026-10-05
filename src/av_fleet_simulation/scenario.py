from .model import TrafficScenario

# A scenario for calibrating the basic model with HDV-only car-following on a
# single-lane 1 km ring. Aim is to match against:
# - vehicle flow as the filter (1800-2400 veh/h/lane).
# - breaking point
# -
# - concave growth pattern - "the standard deviation of the velocity increases in a concave way along vehicle platoon in the empirical oscillations, as observed in the traffic experiments" - Tian, 2016
# TODO: set an initial placement ("even" vs "jam") once TrafficScenario has one;
# random placement puts most runs on the jammed branch.
calibration_hdvs = TrafficScenario(
    rng=42,
    lanes=1,
    road_length=1000,
    torus=True,
    density=20,
    av_share=0.0,
    hdv_alpha=1.0,
)
