import sys
import os
import pytest

# Append src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from LLMEC import LLMEC, joules2kwh

@pytest.mark.parametrize(
    "joules, expected_kwh",
    [
        (3600000, 1),        # Test 1 kWh
        (7200000, 2),        # Test 2 kWh
        (1800000, 0.5),      # Test 0.5 kWh
        (0, 0),              # Test 0 Joules
        (-3600000, -1),      # Test negative values
    ]
)
def test_joules2kwh(joules, expected_kwh):
    """
    Test the joules2kwh function with various inputs.
    """
    result = joules2kwh(joules)
    assert result == pytest.approx(expected_kwh), f"Expected {expected_kwh}, got {result}"

