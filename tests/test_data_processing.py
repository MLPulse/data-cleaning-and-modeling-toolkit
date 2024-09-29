import unittest
import pandas as pd
from src.data_processing import filter_dataframe_by_missing_data

class TestDataProcessing(unittest.TestCase):
    
    def test_filter_dataframe_by_missing_data(self):
        data = {
            'A': [1, 2, None, 4],
            'B': [None, None, 3, 4],
            'C': [5, None, 6, None]
        }
        df = pd.DataFrame(data)
        filtered_df = filter_dataframe_by_missing_data(df, threshold=50)
        self.assertEqual(list(filtered_df.columns), ['A', 'C'])
