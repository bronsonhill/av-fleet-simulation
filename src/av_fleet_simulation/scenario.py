from .model import TrafficScenario

# A scenario for calibrating the basic model with HDV-only car-following on a
# single-lane 1 km ring. Aims are to match against vehicle capacity
# (1800-2400 veh/h/lane). 20 vehicles is 20 veh/km, near the
# flow peak at T = 1.6. Sweep hdv_n over 10-80 for the full fundamental diagram.
# TODO: set an initial placement ("even" vs "jam") once TrafficScenario has one;
# random placement puts most runs on the jammed branch.
calibration_hdvs = TrafficScenario(
    rng=42,
    lanes=1,
    road_length=1000,
    torus=True,
    hdv_n=20,
    hdv_alpha=1.0,
    av1_n=0,
    av2_n=0,
)
