import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add .. to path to import model_selector from parent directory
import sys
sys.path.append('..')
from model_selector import ModelSelector, XGBOOST_AVAILABLE # Assuming model_selector.py is in parent

# Basic dummy data
X_train_dummy = pd.DataFrame(np.random.rand(20, 3), columns=['f1', 'f2', 'f3'])
y_train_dummy_binary = pd.Series(np.random.randint(0, 2, 20))
y_train_dummy_multi = pd.Series(np.random.randint(0, 3, 20))

X_test_dummy = pd.DataFrame(np.random.rand(10, 3), columns=['f1', 'f2', 'f3'])
y_test_dummy_binary = pd.Series(np.random.randint(0, 2, 10))
y_test_dummy_multi = pd.Series(np.random.randint(0, 3, 10))


class TestModelSelector(unittest.TestCase):

    def setUp(self):
        self.selector = ModelSelector()
        # Reduce model set for faster tests if needed, or mock them.
        # For this test, let's limit to a few models to speed it up.
        self.selector.models = {
            'Logistic Regression': self.selector.models['Logistic Regression'],
            'Decision Tree': self.selector.models['Decision Tree'],
        }
        if XGBOOST_AVAILABLE: # Conditionally add XGBoost if it was imported
             self.selector.models['XGBoost'] = ModelSelector().models['XGBoost']


    @patch('model_selector.GridSearchCV')
    def test_evaluate_models_runs(self, mock_grid_search_cv):
        # Mock GridSearchCV to avoid actual training
        mock_estimator = MagicMock()
        mock_estimator.predict.return_value = np.random.randint(0, 2, 10)
        mock_estimator.predict_proba.return_value = np.random.rand(10, 2)

        mock_grid_search_instance = MagicMock()
        mock_grid_search_instance.best_estimator_ = mock_estimator
        mock_grid_search_instance.best_params_ = {'param': 'value'}
        mock_grid_search_cv.return_value = mock_grid_search_instance

        results_df, best_model_obj = self.selector.evaluate_models(
            X_train_dummy, X_test_dummy, y_train_dummy_binary, y_test_dummy_binary
        )

        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertFalse(results_df.empty)
        self.assertIn('model', results_df.columns)
        self.assertIn('f1', results_df.columns)
        self.assertIsNotNone(best_model_obj)
        self.assertIn('name', best_model_obj)
        self.assertIn('model', best_model_obj)

        # Check if GridSearchCV was called for each model
        self.assertEqual(mock_grid_search_cv.call_count, len(self.selector.models))

    # Test actual model fitting for a very small subset to ensure code paths are hit
    # This will be slower.
    def test_evaluate_models_integration_light(self):
        selector_integration = ModelSelector()
        # Use only one simple model for this integration test
        selector_integration.models = {
            'Decision Tree': selector_integration.models['Decision Tree']
        }
        # And very small params for DT
        selector_integration.models['Decision Tree']['params'] = {'max_depth': [2]}


        results_df, best_model_obj = selector_integration.evaluate_models(
            X_train_dummy.iloc[:10], # Smaller data
            X_test_dummy.iloc[:5],
            y_train_dummy_binary.iloc[:10],
            y_test_dummy_binary.iloc[:5]
        )
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), 1) # Only one model tested
        self.assertGreater(results_df['f1'].iloc[0], -1) # F1 score should be calculated
        self.assertIsNotNone(best_model_obj)


    @patch('model_selector.roc_auc_score')
    def test_auc_handling_binary(self, mock_roc_auc):
        mock_roc_auc.return_value = 0.85
        # Setup a model that has predict_proba
        model_with_proba = MagicMock()
        model_with_proba.predict.return_value = y_test_dummy_binary
        model_with_proba.predict_proba.return_value = np.random.rand(len(y_test_dummy_binary), 2) # Binary probas

        # Mock GridSearchCV to return this model
        with patch('model_selector.GridSearchCV') as mock_gscv:
            mock_gscv_instance = MagicMock()
            mock_gscv_instance.best_estimator_ = model_with_proba
            mock_gscv_instance.best_params_ = {}
            mock_gscv.return_value = mock_gscv_instance

            _, best_model_obj = self.selector.evaluate_models(
                X_train_dummy, X_test_dummy, y_train_dummy_binary, y_test_dummy_binary
            )

        # Check that roc_auc_score was called correctly for binary
        # The call inside evaluate_models for binary: roc_auc_score(y_test, y_proba[:, 1])
        # For multiclass: roc_auc_score(y_test, y_proba, multi_class='ovr')
        # Since we use a limited set of models for self.selector, this might only test one path
        # We need to ensure at least one model in self.selector.models has predict_proba
        # Logistic Regression does.

        # Check if roc_auc_score was called.
        # In our mocked setup, all models will use the same mocked best_estimator_
        # So it will be called for each model in self.selector.models
        self.assertTrue(mock_roc_auc.called)
        # Get the arguments of the first call to roc_auc_score
        args, kwargs = mock_roc_auc.call_args_list[0]
        self.assertEqual(args[1].ndim, 1) # For binary, it should be y_proba[:, 1]
        self.assertNotIn('multi_class', kwargs) # No multi_class for binary

    @patch('model_selector.roc_auc_score')
    def test_auc_handling_multiclass(self, mock_roc_auc):
        mock_roc_auc.return_value = 0.75
        model_with_proba = MagicMock()
        model_with_proba.predict.return_value = y_test_dummy_multi
        model_with_proba.predict_proba.return_value = np.random.rand(len(y_test_dummy_multi), 3) # Multiclass probas (3 classes)

        with patch('model_selector.GridSearchCV') as mock_gscv:
            mock_gscv_instance = MagicMock()
            mock_gscv_instance.best_estimator_ = model_with_proba
            mock_gscv_instance.best_params_ = {}
            mock_gscv.return_value = mock_gscv_instance

            # Need to make sure n_classes is correctly identified in evaluate_models
            # y_train_dummy_multi has 3 classes
            _, best_model_obj = self.selector.evaluate_models(
                X_train_dummy, X_test_dummy, y_train_dummy_multi, y_test_dummy_multi
            )

        self.assertTrue(mock_roc_auc.called)
        args, kwargs = mock_roc_auc.call_args_list[0]
        self.assertEqual(args[1].ndim, 2) # For multiclass, it's the full y_proba
        self.assertEqual(kwargs.get('multi_class'), 'ovr')

if __name__ == '__main__':
    unittest.main()
