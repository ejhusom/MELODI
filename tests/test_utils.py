import sys
import os
import pytest

# Append src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import kwh2joules, joules2kwh

@pytest.mark.parametrize(
    "joules, kwh",
    [
        (3600000, 1),        # Test 1 kWh
        (7200000, 2),        # Test 2 kWh
        (1800000, 0.5),      # Test 0.5 kWh
        (0, 0),              # Test 0 Joules
        (-3600000, -1),      # Test negative values
    ]
)
def test_joules2kwh(joules, kwh):
    """
    Test the joules2kwh function with various inputs.
    """
    result = joules2kwh(joules)
    assert result == pytest.approx(kwh), f"Expected {kwh}, got {result}"

@pytest.mark.parametrize(
    "kwh, joules",
    [
        (1, 3600000),        # Test 1 kWh
        (2, 7200000),        # Test 2 kWh
        (0.5, 1800000),      # Test 0.5 kWh
        (0, 0),              # Test 0 kWh
        (-1, -3600000),      # Test negative values
    ]
)
def test_kwh2joules(kwh, joules):
    """
    Test the kwh2joules function with various inputs.
    """
    result = kwh2joules(kwh)
    assert result == pytest.approx(joules), f"Expected {joules}, got {result}"
