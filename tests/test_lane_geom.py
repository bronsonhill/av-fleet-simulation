
from types import SimpleNamespace
import pytest
from av_fleet_simulation.agent import VehicleAgent

class DummyVehicle:
    """ lw vehicle for testing VehicleAgent geom methods w/out running full model"""
    _forward_distance = VehicleAgent._forward_distance
    _overlaps_vehicle = VehicleAgent._overlaps_vehicle
    _target_lane_clear = VehicleAgent._target_lane_clear
    _get_leader = VehicleAgent._get_leader
    _get_follower = VehicleAgent._get_follower

    def __init__(self, x, lane, length=7, torus=True):
        self.position = [x, lane]
        self.length = length
        self.model = SimpleNamespace(scenario=SimpleNamespace(torus=torus))
        self.space = SimpleNamespace(x_min=0, x_max=1000, agents=[])

def make_vehicles(*specs, torus=True):
    """create vehicles sharing one simulated road"""
    vehicles = [DummyVehicle(x, lane, length, torus) for x, lane, length in specs]
    for vehicle in vehicles:
        vehicle.space.agents = vehicles
    return vehicles

def test_01_same_position_different_lanes():
    """geometric overlap ignores lane membership"""
    a, b = make_vehicles((100, 1, 7), (100, 2, 7))
    assert a._overlaps_vehicle(b) is True
    assert a._target_lane_clear(2) is False

def test_02_positive_gap():
    """Vehicles separated by a positive gap do not overlap."""
    a, b = make_vehicles((100, 1, 7), (120, 1, 7))
    assert a._overlaps_vehicle(b) is False
    leader, gap = a._get_leader()
    assert leader is b
    assert gap == 13

def test_03_bumpers_touch():
    """touching bumpers isn't physical overlap"""
    a, b = make_vehicles((100, 1, 7), (107, 1, 7))
    assert a._overlaps_vehicle(b) is False
    leader, gap = a._get_leader()
    assert leader is b
    assert gap == 0

def test_04_toroidal_positive_gap():
    """leader can be across the road boundary"""
    a, b = make_vehicles((990, 1, 7), (5, 1, 7))
    assert a._overlaps_vehicle(b) is False
    leader, gap = a._get_leader()
    assert leader is b
    assert gap == 8

def test_05_toroidal_overlap():
    """overlap across the road boundary is detected"""
    a, b = make_vehicles((998, 1, 7), (3, 1, 7))
    assert a._overlaps_vehicle(b) is True

def test_06_empty_target_lane():
    """changing into an empty lane is geometrically possible"""
    a, b = make_vehicles((100, 1, 7), (150, 1, 7))
    assert a._target_lane_clear(2) is True

def test_07_occupied_target_lane():
    """cvhanging into an overlapping position is rejected"""
    a, b = make_vehicles((100, 1, 7), (103, 2, 7))
    assert a._target_lane_clear(2) is False


"""
> python3 -m uv run pytest tests/test_lane_geom.py -v
all 7 tests passed


"""