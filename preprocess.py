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
        self.numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

    def preprocess(self, df):
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, numeric_features),
                ('cat', self.categorical_transformer, categorical_features)
            ]
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Apply preprocessing
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        return X_train, X_test, y_train, y_test

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
    if STREAMLIT and len(numeric_cols) > 0:
        st.subheader('Outlier Detection (Boxplots)')
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)
    
    # EDA: Class imbalance (if target provided)
    if target_column and target_column in df.columns:
        if STREAMLIT:
            st.subheader('Class Distribution')
            st.bar_chart(df[target_column].value_counts())
        else:
            print('Class Distribution:')
            print(df[target_column].value_counts())
    
    # Preprocessing
    # 1. Impute missing values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            imp = SimpleImputer(strategy='mean')
            df[col] = imp.fit_transform(df[[col]])
        else:
            imp = SimpleImputer(strategy='most_frequent')
            df[col] = imp.fit_transform(df[[col]])
    
    # 2. Encode categorical variables
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in cat_cols:
        cat_cols.remove(target_column)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # 3. Scale numeric features
    scaler = StandardScaler()
    if target_column:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
    else:
        # If no target specified, assume last column
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        target_column = df.columns[-1]
    X_scaled = scaler.fit_transform(X)
    
    # 4. Encode target if categorical
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) < 20 else None
    )
    
    return X_train, X_test, y_train, y_test, target_column
