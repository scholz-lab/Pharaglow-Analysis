import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from pg_analysis.plotter import Worm, UNITS, _lineplot, _hist, _scatter, _heatmap

import matplotlib.pyplot as plt




class TestWormInitialization(unittest.TestCase):
    """Test Worm class initialization."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {
            'frame': [0, 1, 2, 3, 4],
            'x': [10.0, 11.0, 12.0, 13.0, 14.0],
            'y': [20.0, 21.0, 22.0, 23.0, 24.0],
            'pumps': [0, 1, 0, 1, 0]
        }
        self.test_file = os.path.join(self.temp_dir, 'test_0.json')
        
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_worm_initialization_without_load(self):
        """Test Worm initialization without loading data."""
        worm = Worm(
            filename='test_0.json',
            columns=['frame', 'x', 'y'],
            fps=10,
            scale=1.0,
            units=UNITS,
            load=False
        )
        self.assertEqual(worm.fps, 10)
        self.assertEqual(worm.scale, 1.0)
        self.assertEqual(worm.particle_index, 0)


class TestWormGetMetric(unittest.TestCase):
    """Test Worm.get_metric method."""
    
    def setUp(self):
        """Create a Worm instance with sample data."""
        self.worm = Worm(
            filename='test_0.json',
            columns=['frame', 'x', 'y', 'pumps'],
            fps=10,
            scale=1.0,
            units=UNITS,
            load=False
        )
        self.worm.data = pd.DataFrame({
            'frame': [0, 1, 2, 3, 4],
            'x': [10.0, 11.0, 12.0, 13.0, 14.0],
            'y': [20.0, 21.0, 22.0, 23.0, 24.0],
            'pumps': [0, 1, 0, 1, 0]
        })
    
    def test_get_metric_mean(self):
        """Test getting mean metric."""
        mean_val = self.worm.get_metric('pumps', 'mean')
        self.assertEqual(mean_val, 0.4)
    
    def test_get_metric_sum(self):
        """Test getting sum metric."""
        sum_val = self.worm.get_metric('pumps', 'sum')
        self.assertEqual(sum_val, 2)
    
    def test_get_metric_count(self):
        """Test getting count metric."""
        count_val = self.worm.get_metric('pumps', 'N')
        self.assertEqual(count_val, 5)
    
    def test_get_metric_invalid_key(self):
        """Test get_metric with invalid key."""
        with self.assertRaises(AssertionError):
            self.worm.get_metric('nonexistent', 'mean')
    
    def test_get_metric_invalid_metric(self):
        """Test get_metric with invalid metric."""
        with self.assertRaises(Exception):
            self.worm.get_metric('pumps', 'invalid_metric')


class TestWormAddColumn(unittest.TestCase):
    """Test Worm.add_column method."""
    
    def setUp(self):
        """Create a Worm instance with sample data."""
        self.worm = Worm(
            filename='test_0.json',
            columns=['frame', 'x', 'y'],
            fps=10,
            scale=1.0,
            units=UNITS,
            load=False
        )
        self.worm.data = pd.DataFrame({
            'frame': [0, 1, 2, 3, 4],
            'x': [10.0, 11.0, 12.0, 13.0, 14.0],
            'y': [20.0, 21.0, 22.0, 23.0, 24.0]
        })
    
    def test_add_column_new(self):
        """Test adding a new column."""
        new_values = [1, 2, 3, 4, 5]
        self.worm.add_column('test_col', new_values)
        self.assertIn('test_col', self.worm.data.columns)
        self.assertTrue((self.worm.data['test_col'] == new_values).all())
    
    def test_add_column_length_mismatch(self):
        """Test adding a column with mismatched length."""
        with self.assertWarns(UserWarning):
            self.worm.add_column('test_col', [1, 2, 3])


class TestWormGetData(unittest.TestCase):
    """Test Worm.get_data method."""
    
    def setUp(self):
        """Create a Worm instance with sample data."""
        self.worm = Worm(
            filename='test_0.json',
            columns=['frame', 'x', 'y'],
            fps=10,
            scale=1.0,
            units=UNITS,
            load=False
        )
        self.worm.data = pd.DataFrame({
            'frame': [0, 1, 2, 3, 4],
            'x': [10.0, 11.0, 12.0, 13.0, 14.0],
            'y': [20.0, 21.0, 22.0, 23.0, 24.0]
        })
    
    def test_get_data_all(self):
        """Test getting all data."""
        data = self.worm.get_data()
        self.assertIsInstance(data, pd.DataFrame)
    
    def test_get_data_single_column(self):
        """Test getting a single column."""
        data = self.worm.get_data('x')
        self.assertIsInstance(data, pd.Series)
        self.assertEqual(len(data), 5)


class TestPlottingFunctions(unittest.TestCase):
    """Test plotting utility functions."""
    
    def setUp(self):
        """Create sample data for plotting."""
        self.x = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        self.y = pd.DataFrame([[2, 4], [6, 8], [10, 12]])
        self.fig, self.ax = plt.subplots()
    
    def tearDown(self):
        """Clean up plots."""
        plt.close(self.fig)
    
    def test_lineplot_single_axis(self):
        """Test _lineplot with single axis."""
        plot = _lineplot(self.x, self.y, None, self.ax)
        self.assertIsNotNone(plot)
    
    def test_hist_single_axis(self):
        """Test _hist with single axis."""
        plot = _hist(self.y, self.ax)
        self.assertIsNotNone(plot)
    
    def test_scatter_single_axis(self):
        """Test _scatter with single axis."""
        plot = _scatter(self.x, self.y, None, None, self.ax)
        self.assertIsNotNone(plot)


class TestCalculateProperties(unittest.TestCase):
    """Test Worm.calculate_properties method."""
    
    def setUp(self):
        """Create a Worm instance with sample data."""
        self.worm = Worm(
            filename='test_0.json',
            columns=['frame', 'x', 'y'],
            fps=10,
            scale=1.0,
            units=UNITS,
            load=False
        )
        self.worm.data = pd.DataFrame({
            'frame': [0, 1, 2, 3, 4],
            'x': [10.0, 11.0, 12.0, 13.0, 14.0],
            'y': [20.0, 21.0, 22.0, 23.0, 24.0],
            'pumps': [0, 1, 0, 1, 0]
        })
    
    def test_calculate_properties(self):
        """Test calculating properties."""
        for key in ['time', 'locations', 'velocity', 'reversals']:
            self.worm.calculate_property(key)
        self.worm.calculate_property('smoothed', key='x', window=3)
        self.worm.calculate_property('count_rate', key='pumps', window=2)
        self.assertIn('time', self.worm.data.columns)
        self.assertIn('x_smoothed', self.worm.data.columns)
        self.assertIn('count_rate_pumps', self.worm.data.columns)
        self.assertIn('reversals', self.worm.data.columns)


if __name__ == '__main__':
    unittest.main()