import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Try to import streamlit, fallback to CLI if not available
try:
    import streamlit as st
    STREAMLIT = True
except ImportError:
    STREAMLIT = False

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = None  # To store the fitted LabelEncoder
        self.numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

    def preprocess(self, df, target_column):
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Handle target encoding if categorical BEFORE splitting
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # Identify numeric and categorical columns from X
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Split data - stratify if possible (classification)
        stratify_option = None
        if y.nunique() < 20 and y.nunique() > 1: # Heuristic for classification tasks
            stratify_option = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_option
        )

        # Create preprocessing pipeline for features
        # Ensure transformers list is correctly constructed based on feature presence
        transformers = []
        if not numeric_features.empty:
            transformers.append(('num', self.numeric_transformer, numeric_features))
        if not categorical_features.empty:
            transformers.append(('cat', self.categorical_transformer, categorical_features))

        if not transformers:
            # Handle case with no numeric or categorical features (e.g., all features are of a different type not handled)
            # Or if X_train is empty
            if X_train.empty:
                 return X_train.copy(), X_test.copy(), y_train, y_test # Or raise error
            # If X_train is not empty but no transformers, return as is (or handle as error)
            # For now, assume this implies no preprocessing needed for features if they are not num/cat
            return X_train.values, X_test.values, y_train, y_test


        preprocessor = ColumnTransformer(transformers=transformers)

        # Apply preprocessing to features
        # Fit on X_train and transform X_train
        X_train_processed = preprocessor.fit_transform(X_train)
        # Transform X_test using the preprocessor fitted on X_train
        X_test_processed = preprocessor.transform(X_test)

        # If after processing, the output is sparse and we want dense, convert it
        # This is common with OneHotEncoder. For now, let's assume sparse is fine.
        # if hasattr(X_train_processed, "toarray"):
        # X_train_processed = X_train_processed.toarray()
        # X_test_processed = X_test_processed.toarray()

        return X_train_processed, X_test_processed, y_train, y_test

def run_preprocessing(uploaded_file, target_column=None, test_size=0.2, random_state=42):
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    # EDA: Missing values
    missing = df.isnull().sum()
    if STREAMLIT:
        st.subheader('Missing Values')
        st.write(missing[missing > 0])
    else:
        print('Missing Values:')
        print(missing[missing > 0])

    # EDA: Distributions
    if STREAMLIT:
        st.subheader('Feature Distributions')
        st.write(df.describe())
        st.write(df.head())
    else:
        print(df.describe())
        print(df.head())

    # EDA: Outliers (boxplot for numeric columns)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_column in numeric_cols: # Exclude target column from feature boxplots if it's numeric
        numeric_cols.remove(target_column)

    if STREAMLIT and len(numeric_cols) > 0:
        st.subheader('Outlier Detection (Boxplots for Features)')
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax) # Use original df for EDA
            st.pyplot(fig)
    
    # EDA: Class imbalance (if target provided and in df)
    if target_column and target_column in df.columns:
        if STREAMLIT:
            st.subheader(f'Distribution of Target Column: {target_column}')
            # Check if target is categorical or numeric for appropriate chart
            if df[target_column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[target_column]) or df[target_column].nunique() < 20: # Heuristic for categorical-like
                st.bar_chart(df[target_column].value_counts())
            else: # For continuous target, show histogram
                fig, ax = plt.subplots()
                sns.histplot(df[target_column], kde=True, ax=ax)
                st.pyplot(fig)
        else:
            print(f'Distribution of Target Column: {target_column}')
            print(df[target_column].value_counts())
    else:
        if STREAMLIT:
            st.warning("Target column not found in the dataframe. Skipping target-specific EDA.")
        else:
            print("Warning: Target column not found. Skipping target-specific EDA.")
        # Decide how to handle if target_column is mandatory. For now, assume it might not be found.
        # Or, ensure target_column is always valid before this point.
        # If no target column, preprocessing might be limited or need a different approach.
        # For this refactor, we assume target_column is provided and valid for the DataPreprocessor.
        # If not, DataPreprocessor might fail. Consider adding a check before instantiating/calling it.
        # However, the original function also implies a target_column is expected.

    # Instantiate DataPreprocessor
    preprocessor_instance = DataPreprocessor()

    # Perform preprocessing using the DataPreprocessor class
    # Note: test_size and random_state from run_preprocessing are not directly passed to
    # preprocessor_instance.preprocess as it uses its own internal values (0.2, 42).
    # This could be made configurable in DataPreprocessor if needed.
    if target_column and target_column in df.columns:
        X_train, X_test, y_train, y_test = preprocessor_instance.preprocess(df, target_column)
    else:
        # Handle case where target column is not valid or not in df for actual preprocessing
        # This might mean returning empty/None dataframes or raising an error
        if STREAMLIT:
            st.error(f"Target column '{target_column}' not found in the uploaded file. Cannot proceed with preprocessing.")
        else:
            print(f"Error: Target column '{target_column}' not found. Cannot proceed.")
        # Return empty DataFrames or None to signify failure or an unusable state
        # The number of return values should match the function signature
        empty_df_processed = pd.DataFrame() # Or np.array([])
        return empty_df_processed, empty_df_processed, pd.Series(dtype='object'), pd.Series(dtype='object'), target_column


    # The old manual preprocessing steps (imputation, encoding, scaling, splitting) are now handled by DataPreprocessor.
    
    return X_train, X_test, y_train, y_test, target_column
