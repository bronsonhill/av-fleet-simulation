import random
from types import SimpleNamespace

import pytest

from av_fleet_simulation.agent import VehicleAgent
from av_fleet_simulation.model import MultiFleetTrafficModel


class DummyVehicle:
    """lw vehicle for testing VehicleAgent geom methods w/out running full model"""

    _forward_distance = VehicleAgent._forward_distance
    _overlaps_vehicle = VehicleAgent._overlaps_vehicle

    def __init__(
        self,
        x: float,
        lane: int,
        speed: float,
        target_lane: int | None,
        length: int = 7,
    ):
        self.position = [x, lane]
        self.v = speed
        self.target_lane = target_lane
        self.length = length
        self.model = SimpleNamespace(scenario=SimpleNamespace(torus=True))
        self.space = SimpleNamespace(x_min=0, x_max=1000)


def resolve(*vehicles, seed=42):
    "run conflict resolve w/lw model object"
    model = SimpleNamespace(agents=list(vehicles), random=random.Random(seed))
    MultiFleetTrafficModel._resolve_lc_conflicts(model)


def test_01_single_proposal_approved():
    """unoccupied target position is approved"""
    a = DummyVehicle(x=100, lane=1, speed=20, target_lane=2)
    resolve(a)
    assert a.target_lane == 2


def test_02_faster_vehicle_gets_priority():
    """ewhen two vehicles propose overlapping target positions faster vehicle receives priority"""
    a = DummyVehicle(x=100, lane=1, speed=20, target_lane=2)
    b = DummyVehicle(x=102, lane=3, speed=15, target_lane=2)
    resolve(a, b)
    assert a.target_lane == 2
    assert b.target_lane is None


def test_03_non_overlapping_proposals_approved():
    """two vehicles entering the same lane at sufficiently separated positions should both be approved"""
    a = DummyVehicle(x=100, lane=1, speed=20, target_lane=2)
    b = DummyVehicle(x=150, lane=3, speed=15, target_lane=2)
    resolve(a, b)
    assert a.target_lane == 2
    assert b.target_lane == 2


def test_04_occupied_target_position_rejected():
    """vehicle cannot enter a position occupied by another vehicle that's staying in its lane"""
    a = DummyVehicle(x=100, lane=1, speed=20, target_lane=2)
    b = DummyVehicle(x=102, lane=2, speed=15, target_lane=None)
    resolve(a, b)
    assert a.target_lane is None
    assert b.target_lane is None


def test_05_simultaneous_swap_rejected():
    """under conservative reservation rule vehicles can't swap overlapping positions"""
    a = DummyVehicle(x=100, lane=1, speed=20, target_lane=2)
    b = DummyVehicle(x=100, lane=2, speed=15, target_lane=1)
    resolve(a, b)
    assert a.target_lane is None
    assert b.target_lane is None


"""
> python3 -m uv run pytest tests/test_lc_conflict_resolution.py -v
all 5 tests passed
"""
