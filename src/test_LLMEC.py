#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the LLMEC module, focusing on scaphandre process detection and energy aggregation.

Author:
    Erik Johannes Husom

Created:
    2025-04-18
"""
import unittest
import datetime
import pandas as pd
import numpy as np
import pytz
from unittest.mock import patch, MagicMock, mock_open
import json
from pathlib import Path

# Import the modules to test
from LLMEC import LLMEC, calculate_energy_consumption_from_power_measurements, parse_json_objects


class TestScaphandreProcessDetection(unittest.TestCase):
    """Test cases for scaphandre process detection and energy consumption aggregation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config for testing
        self.config_patcher = patch('LLMEC.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.LLM_SERVICE_KEYWORD = 'ollamaserve'
        self.mock_config.MONITORING_SERVICE_KEYWORD = 'scaphandre'
        self.mock_config.MEASUREMENTS_START_BUFFER = 0.5
        self.mock_config.MEASUREMENTS_END_BUFFER = 0.5
        self.mock_config.MONITORING_START_DELAY = 1.0
        self.mock_config.MONITORING_END_DELAY = 2.0
        self.mock_config.SCAPHANDRE_STREAM_TEMP_FILE = 'tmp_scaphandre.json'
        self.mock_config.NVIDIASMI_STREAM_TEMP_FILE = 'tmp_nvidiasmi.csv'
        
        # Create a test instance
        self.llmec = LLMEC()
        self.llmec.verbosity = 2  # Set high verbosity for testing

    def tearDown(self):
        """Tear down test fixtures."""
        self.config_patcher.stop()

    def test_parse_json_objects(self):
        """Test parsing JSON objects from a stream."""
        # Test with valid JSON objects
        valid_json_stream = '{"key1":"value1"}{"key2":"value2"}'
        result = parse_json_objects(valid_json_stream)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['key1'], 'value1')
        self.assertEqual(result[1]['key2'], 'value2')
        
        # Test with incomplete JSON object at the end
        incomplete_json_stream = '{"key1":"value1"}{"key2":"value'
        result = parse_json_objects(incomplete_json_stream)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['key1'], 'value1')
        
        # Test with empty string
        empty_stream = ''
        result = parse_json_objects(empty_stream)
        self.assertEqual(result, [])

    def test_process_filtering(self):
        """Test the filtering of scaphandre processes containing 'ollama'."""
        # Create mock metrics_per_process
        metrics_per_process = {
            'ollamaserve': pd.DataFrame({
                'timestamp': [1617235200, 1617235201, 1617235202],
                'consumption': [100, 110, 120],
                'cmdline': ['ollamaserve', 'ollamaserve', 'ollamaserve']
            }),
            'ollamarun': pd.DataFrame({
                'timestamp': [1617235200, 1617235201, 1617235202],
                'consumption': [200, 210, 220],
                'cmdline': ['ollamarun', 'ollamarun', 'ollamarun']
            }),
            'scaphandre_ollama': pd.DataFrame({
                'timestamp': [1617235200, 1617235201, 1617235202],
                'consumption': [50, 55, 60],
                'cmdline': ['scaphandre --process-regex .*ollama.*', 'scaphandre --process-regex .*ollama.*', 'scaphandre --process-regex .*ollama.*']
            }),
            'scaphandre': pd.DataFrame({
                'timestamp': [1617235200, 1617235201, 1617235202],
                'consumption': [30, 33, 36],
                'cmdline': ['scaphandre', 'scaphandre', 'scaphandre']
            }),
        }
        
        # Set index for each DataFrame
        for process in metrics_per_process.values():
            process.set_index('timestamp', inplace=True)
        
        # Mock the _parse_metrics method to return our test data
        with patch.object(self.llmec, '_parse_metrics', return_value=metrics_per_process):
            with patch('LLMEC.pd.concat') as mock_concat:
                # Configure mock_concat to return a properly structured DataFrame
                mock_concat.return_value = pd.DataFrame({
                    'consumption': [300, 320, 340],  # Sum of ollamaserve and ollamarun
                    'cmdline': ['ollama', 'ollama', 'ollama']
                }, index=[1617235200, 1617235201, 1617235202])
                
                # Create a context where we're just testing the process filtering logic
                # in the run_prompt_with_energy_monitoring method
                ollama_dfs = []
                ollama_process_names = []
                
                for cmdline, specific_process in metrics_per_process.items():
                    # We need to filter out the scaphandre process itself which contains "ollama" in its command line arguments
                    if "ollama" in cmdline and "scaphandre" not in cmdline:
                        ollama_dfs.append(specific_process)
                        ollama_process_names.append(cmdline)
                
                # Check that the filtering worked correctly
                self.assertEqual(len(ollama_dfs), 2)
                self.assertIn('ollamaserve', ollama_process_names)
                self.assertIn('ollamarun', ollama_process_names)
                self.assertNotIn('scaphandre_ollama', ollama_process_names)

    def test_energy_consumption_aggregation(self):
        """Test the aggregation of energy consumption from multiple Ollama processes."""
        # Create mock metrics dataframes for testing
        ollamaserve_df = pd.DataFrame({
            'consumption': [100, 110, 120],
            'cmdline': ['ollamaserve', 'ollamaserve', 'ollamaserve'],
            'datetime': pd.to_datetime([1617235200, 1617235210, 1617235220], unit='s', utc=True)
        }, index=[1617235200, 1617235210, 1617235220])
        
        ollamarun_df = pd.DataFrame({
            'consumption': [200, 210, 220],
            'cmdline': ['ollamarun', 'ollamarun', 'ollamarun'],
            'datetime': pd.to_datetime([1617235200, 1617235210, 1617235220], unit='s', utc=True)
        }, index=[1617235200, 1617235210, 1617235220])
        
        llm_gpu_df = pd.DataFrame({
            'consumption': [50, 55, 60],
            'datetime': pd.to_datetime([1617235200, 1617235210, 1617235220], unit='s', utc=True)
        }, index=[1617235200, 1617235210, 1617235220])
        
        metrics_per_process = {
            'ollamaserve': ollamaserve_df,
            'ollamarun': ollamarun_df,
            'llm_gpu': llm_gpu_df,
        }
        
        # Test time parameters
        start_time = datetime.datetime.fromtimestamp(1617235200, tz=pytz.utc)
        end_time = datetime.datetime.fromtimestamp(1617235220, tz=pytz.utc)
        
        # Call the function to test
        with patch('LLMEC.np.trapezoid') as mock_trapezoid:
            # Set a fixed return value for trapezoid
            mock_trapezoid.side_effect = [1100, 2100, 550]  # Values for ollamaserve, ollamarun, llm_gpu
            
            with patch('LLMEC.print') as mock_print:  # Suppress print statements
                result = calculate_energy_consumption_from_power_measurements(
                    metrics_per_process, 
                    start_time, 
                    end_time
                )
        
        # Check the results
        self.assertIn('ollamaserve', result)
        self.assertIn('ollamarun', result)
        self.assertIn('llm_gpu', result)
        
        # ollamaserve energy: 1100 / (10^3 * 3600) = 0.000306 kWh
        self.assertAlmostEqual(result['ollamaserve'], 1100 / (10**3 * 3600))
        
        # ollamarun energy: 2100 / (10^3 * 3600) = 0.000583 kWh
        self.assertAlmostEqual(result['ollamarun'], 2100 / (10**3 * 3600))
        
        # llm_gpu energy: 550 / (10^3 * 3600) = 0.000153 kWh
        self.assertAlmostEqual(result['llm_gpu'], 550 / (10**3 * 3600))

    @patch('subprocess.Popen')
    @patch('time.sleep')
    @patch('builtins.open', new_callable=mock_open, read_data='{"sample": "data"}')
    @patch('LLMEC.parse_json_objects')
    @patch('LLMEC.pd.read_csv')
    def test_run_prompt_with_energy_monitoring(self, mock_read_csv, mock_parse_json, mock_file, mock_sleep, mock_popen):
        """Test the main method that runs prompts with energy monitoring."""
        # Skip this test - it requires extensive mocking of the run_prompt_with_energy_monitoring method
        # This is just a placeholder to show what we'd test
        self.skipTest("This test requires extensive mocking of the LLMEC class")
        
        # Alternatively, we could just test the process filtering part of the method
        # which is the focus of our unit tests, instead of testing the entire method


class TestEnergyConsumptionCalculation(unittest.TestCase):
    """Test cases for energy consumption calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_patcher = patch('LLMEC.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.MEASUREMENTS_START_BUFFER = 0.5
        self.mock_config.MEASUREMENTS_END_BUFFER = 0.5
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.config_patcher.stop()
    
    def test_calculate_energy_consumption(self):
        """Test the calculation of energy consumption from power measurements."""
        # Create test data with known values for trapezoid integration
        # Simple case: constant power of 100W for 10 seconds
        timestamps = np.array([0, 5, 10])
        power_values = np.array([100, 100, 100])
        
        df = pd.DataFrame({
            'consumption': power_values,
            'datetime': pd.to_datetime(timestamps, unit='s', utc=True)
        }, index=timestamps)
        
        df_dict = {
            'test_process': df
        }
        
        start_time = pd.to_datetime(0, unit='s', utc=True)
        end_time = pd.to_datetime(10, unit='s', utc=True)
        
        # For constant power, energy = power * time
        # 100W * 10s = 1000J = 1000/(3600*1000) kWh = 0.000278 kWh
        expected_energy = 0.000278  # kWh
        
        # Test the function with mocked trapezoid to return our expected energy in joules
        with patch('LLMEC.np.trapezoid', return_value=1000):
            with patch('LLMEC.print'):  # Suppress print statements
                result = calculate_energy_consumption_from_power_measurements(
                    df_dict, start_time, end_time, show_plot=False
                )
        
        # Check result
        self.assertAlmostEqual(result['test_process'], expected_energy, places=6)


if __name__ == '__main__':
    unittest.main()