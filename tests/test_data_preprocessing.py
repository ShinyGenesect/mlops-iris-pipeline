import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor:

    def setup_method(self):
        """Setup for each test method"""
        self.preprocessor = DataPreprocessor()

    def test_load_data(self):
        """Test data loading functionality"""
        df = self.preprocessor.load_data()

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 150  # Iris dataset has 150 samples
        assert df.shape[1] == 5    # 4 features + 1 target

        # Check column names
        expected_columns = ['sepal length (cm)', 'sepal width (cm)',
                          'petal length (cm)', 'petal width (cm)', 'target']
        assert list(df.columns) == expected_columns

        # Check target values
        assert set(df['target'].unique()) == {0, 1, 2}

        # Check data types
        assert df['target'].dtype in ['int64', 'int32']
        for col in df.columns[:-1]:  # All except target
            assert df[col].dtype in ['float64', 'float32']

    def test_preprocess_data(self):
        """Test data preprocessing functionality"""
        df = self.preprocessor.load_data()
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(df)

        # Check shapes
        assert X_train.shape[0] == 120  # 80% of 150
        assert X_test.shape[0] == 30    # 20% of 150
        assert X_train.shape[1] == 4    # 4 features
        assert X_test.shape[1] == 4     # 4 features

        # Check that data is scaled (mean ~ 0, std ~ 1)
        assert abs(np.mean(X_train)) < 0.1
        assert abs(np.std(X_train) - 1.0) < 0.1

        # Check stratification (each class represented in train/test)
        assert len(np.unique(y_train)) == 3
        assert len(np.unique(y_test)) == 3

    def test_transform_new_data(self):
        """Test transformation of new data"""
        # First fit the scaler
        df = self.preprocessor.load_data()
        self.preprocessor.preprocess_data(df)

        # Test with dictionary input
        new_data_dict = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }

        transformed = self.preprocessor.transform_new_data(new_data_dict)
        assert transformed.shape == (1, 4)

        # Test with DataFrame input
        new_data_df = pd.DataFrame([new_data_dict])
        transformed_df = self.preprocessor.transform_new_data(new_data_df)
        assert transformed_df.shape == (1, 4)

        # Results should be the same
        np.testing.assert_array_equal(transformed, transformed_df)

    def test_save_load_preprocessor(self, tmp_path):
        """Test saving and loading preprocessor"""
        # Fit the preprocessor
        df = self.preprocessor.load_data()
        self.preprocessor.preprocess_data(df)

        # Save preprocessor
        filepath = tmp_path / "test_scaler.pkl"
        self.preprocessor.save_preprocessor(str(filepath))

        # Create new preprocessor and load
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load_preprocessor(str(filepath))

        # Test that they produce same results
        test_data = pd.DataFrame([{
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }])

        original_result = self.preprocessor.transform_new_data(test_data)
        loaded_result = new_preprocessor.transform_new_data(test_data)

        np.testing.assert_array_almost_equal(original_result, loaded_result)
