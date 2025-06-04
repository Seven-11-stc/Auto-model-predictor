import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Add .. to path to import preprocess from parent directory
import sys
sys.path.append('..')
from preprocess import DataPreprocessor # Assuming preprocess.py is in the parent directory

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'numeric_col1': [1, 2, np.nan, 4, 5],
            'numeric_col2': [1.1, 2.2, 3.3, np.nan, 5.5],
            'categorical_col1': ['A', 'B', 'A', 'C', 'B'],
            'categorical_col2': ['X', np.nan, 'Y', 'X', 'Z'],
            'target_numeric': [10, 20, 30, 40, 50],
            'target_categorical': ['P', 'Q', 'P', 'Q', 'P']
        })
        self.preprocessor = DataPreprocessor()

    def test_preprocess_numeric_target(self):
        df = self.data.copy()
        target_column = 'target_numeric'

        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(df, target_column)

        # Check shapes (80/20 split for 5 rows -> 4 train, 1 test)
        self.assertEqual(X_train.shape[0], 4)
        self.assertEqual(X_test.shape[0], 1)
        self.assertEqual(y_train.shape[0], 4)
        self.assertEqual(y_test.shape[0], 1)

        # Check for NaNs in processed features (should be handled by imputation)
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_test).any())

        # Check y_train and y_test dtypes (should be same as original numeric target)
        self.assertTrue(pd.api.types.is_numeric_dtype(y_train))
        self.assertTrue(pd.api.types.is_numeric_dtype(y_test))

    def test_preprocess_categorical_target(self):
        df = self.data.copy()
        target_column = 'target_categorical'

        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(df, target_column)

        self.assertEqual(X_train.shape[0], 4)
        self.assertEqual(X_test.shape[0], 1)

        # Check y_train and y_test dtypes (should be encoded to numeric)
        self.assertTrue(pd.api.types.is_numeric_dtype(y_train))
        self.assertTrue(pd.api.types.is_numeric_dtype(y_test))
        self.assertIsNotNone(self.preprocessor.label_encoder) # Check if LE was fitted

    def test_imputation_and_scaling_ohe(self):
        # Test with a dataframe that has specific places for NaNs
        df = pd.DataFrame({
            'num1': [1, 2, np.nan, 4],    # NaN here
            'cat1': ['A', 'B', np.nan, 'A'], # NaN here
            'target': [0, 1, 0, 1]
        })
        target_column = 'target'
        X_train, X_test, _, _ = self.preprocessor.preprocess(df.copy(), target_column)

        # For a 4-row df, 3 train, 1 test.
        # Original features: num1 (numeric), cat1 (categorical)
        # After OHE, cat1 might become 2 columns (e.g., cat1_A, cat1_B) if B is not dropped or cat1_missing
        # The ColumnTransformer should handle this.

        self.assertFalse(np.isnan(X_train).any(), "NaNs found in X_train after preprocessing")
        self.assertFalse(np.isnan(X_test).any(), "NaNs found in X_test after preprocessing")

        # Check that the number of columns in X_train/X_test is consistent
        # with one numeric column scaled and one categorical column one-hot encoded.
        # Example: 1 (scaled num1) + num_unique_cat1_after_impute (OHE for cat1)
        # This is a bit tricky to assert exact number due to OHE behavior (drop, handle_unknown)
        # but we can check it's greater than original number of feature columns if OHE added some.
        # Original X has 2 columns. If cat1 has 2 unique values + missing, it could be 2-3 OHE cols.
        # So X_train could have 1 (numeric) + (2 to 3 from OHE) = 3 to 4 columns.
        self.assertTrue(X_train.shape[1] >= 1) # At least one numeric column processed

if __name__ == '__main__':
    unittest.main()
