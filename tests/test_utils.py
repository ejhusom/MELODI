import sys
import os
import pytest

# Append src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import utils

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
    result = utils.joules2kwh(joules)
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
    result = utils.kwh2joules(kwh)
    assert result == pytest.approx(joules), f"Expected {joules}, got {result}"

def test_clean_filename():
    # Test replacement of unsafe characters
    assert utils.clean_filename('my:file/name*?.txt') == 'my_file_name__.txt'

    # Test replacement of multiple unsafe characters
    assert utils.clean_filename('a/b:c?d*e"f|g') == 'a_b_c_d_e_f_g'
    
    # Test leading/trailing whitespace stripping
    assert utils.clean_filename('  myfile.txt  ') == 'myfile.txt'
    
    # Test no illegal characters - filename should remain the same
    assert utils.clean_filename('safe_filename.txt') == 'safe_filename.txt'
    
    # Test reserved Windows filename
    assert utils.clean_filename('CON.txt') == 'CON.txt'

    # Test filename without extension
    assert utils.clean_filename('test_file') == 'test_file'

    # Test empty string as input
    assert utils.clean_filename('') == ''
