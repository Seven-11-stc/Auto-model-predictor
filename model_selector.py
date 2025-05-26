from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

# Add XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class ModelSelector:
    def __init__(self):
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression,
                'params': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier,
                'params': {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 10]}
            },
            'Random Forest': {
                'model': RandomForestClassifier,
                'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier,
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            },
            'SVM': {
                'model': SVC,
                'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'probability': [True]}
            },
            'KNN': {
                'model': KNeighborsClassifier,
                'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            },
            'Naive Bayes': {
                'model': GaussianNB,
                'params': {}  # No parameters to tune
            },
            'Neural Network': {
                'model': MLPClassifier,
                'params': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh'], 'max_iter': [200]}
            }
        }
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = {
                'model': XGBClassifier,
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'use_label_encoder': [False], 'eval_metric': ['mlogloss']}
            }

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        results = []
        best_model = None
        best_f1 = -1
        best_model_name = None
        best_model_obj = None
        n_classes = len(np.unique(y_train))
        for name, config in self.models.items():
            model = config['model']()
            # Perform grid search
            grid_search = GridSearchCV(
                model, config['params'], scoring='f1_weighted', cv=3, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            # Get best model
            best = grid_search.best_estimator_
            y_pred = best.predict(X_test)
            # AUC handling for binary/multiclass
            try:
                if hasattr(best, "predict_proba"):
                    y_proba = best.predict_proba(X_test)
                    if n_classes == 2:
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                else:
                    auc = np.nan
            except Exception:
                auc = np.nan
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'roc_auc': auc
            }
            results.append({
                'model': name,
                'best_params': grid_search.best_params_,
                **metrics
            })
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_model = best
                best_model_name = name
                best_model_obj = {
                    'name': name,
                    'model': best,
                    'metrics': metrics,
                    'params': grid_search.best_params_
                }
        return pd.DataFrame(results), best_model_obj

def run_model_selection(X_train, X_test, y_train, y_test):
    selector = ModelSelector()
    results_df, best_model_obj = selector.evaluate_models(X_train, X_test, y_train, y_test)
    return results_df, best_model_obj
