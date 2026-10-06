from .model import TrafficScenario

# A scenario for calibrating the basic model with HDV-only car-following on a
# single-lane 1 km ring. Patterns to match:
# - capacity: 2400 pc/h/ln, from c = min(2200 + 10(FFS - 50), 2400)
#   at FFS ~74 mi/h; HCM 6th ed. (TRB 2016), Ch. 12. Tolerance chosen: ± 150.
# - density at capacity (held out): the flow peak should sit at 45 pc/mi/ln
#   ~ 28 veh/km/ln; HCM 6th ed., Ch. 12. Tolerance chosen: ± 3.
# - concave growth (held out): speed std rises along the platoon, concavely;
#   Jiang et al. (2014), Tian et al. (2016).
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
