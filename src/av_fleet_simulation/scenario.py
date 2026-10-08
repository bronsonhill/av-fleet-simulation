from .model import TrafficScenario

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

# A scenario for the concave-growth test: 26 HDVs start in a jam on a single
# lane; the experiment pins the front car to a fixed speed. The ring is long
# enough that the platoon never spreads round to the leader's back.
# Jiang et al. (2014); Tian et al. (2016), Sec. 3.3.
platoon_hdvs = TrafficScenario(
    rng=42,
    lanes=1,
    road_length=5000,
    torus=True,
    placement="jam",
    density=5.2,  # 26 vehicles on 5 km
    av_share=0.0,
    hdv_alpha=1.0,
)
