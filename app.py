import streamlit as st
import pandas as pd
import numpy as np
from preprocess import run_preprocessing
from model_selector import run_model_selection
from utils import plot_metrics, plot_confusion_matrix, plot_roc_curve
import os
import joblib

def main():
    st.title("AutoML Model Recommender")
    st.write("Upload your dataset and let the app find the best model for you!")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    target_column = st.text_input("Enter the target column name (case-sensitive):")
    if uploaded_file is not None and target_column:
        # EDA & Preprocessing
        if st.button("Run EDA & Preprocessing"):
            with st.spinner("Running EDA and preprocessing..."):
                X_train, X_test, y_train, y_test, target = run_preprocessing(uploaded_file, target_column)
                st.success("EDA and preprocessing complete!")
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['target'] = target

    # Model selection and evaluation
    if (
        'X_train' in st.session_state and
        st.button("Train and Evaluate Models")
    ):
        with st.spinner("Training and evaluating models..."):
            results_df, best_model_obj = run_model_selection(
                st.session_state['X_train'],
                st.session_state['X_test'],
                st.session_state['y_train'],
                st.session_state['y_test']
            )
            st.write("# Model Comparison")
            st.dataframe(results_df)
            plot_metrics(results_df)
            st.write(f"## Best Model: {best_model_obj['name']}")
            st.write(f"Best F1 Score: {best_model_obj['metrics']['f1']:.4f}")
            # Save best model
            if not os.path.exists("models"):
                os.makedirs("models")
            model_path = f"models/best_model_{best_model_obj['name'].replace(' ', '_')}.pkl"
            joblib.dump(best_model_obj['model'], model_path)
            st.success(f"Best model saved as {model_path}!")
            # Download link
            with open(model_path, "rb") as f:
                st.download_button(
                    label="Download Best Model",
                    data=f,
                    file_name=os.path.basename(model_path),
                    mime="application/octet-stream"
                )
            # Confusion matrix and ROC curve
            y_pred = best_model_obj['model'].predict(st.session_state['X_test'])
            if hasattr(best_model_obj['model'], 'predict_proba'):
                y_proba = best_model_obj['model'].predict_proba(st.session_state['X_test'])
            else:
                y_proba = None
            labels = np.unique(st.session_state['y_test'])
            st.write("### Confusion Matrix")
            cm_fig = plot_confusion_matrix(st.session_state['y_test'], y_pred, labels)
            st.pyplot(cm_fig) # Display the captured figure
            if y_proba is not None:
                st.write("### ROC Curve")
                # Ensure labels are suitable for predict_proba output (typically 0, 1, ... n_classes-1)
                # np.unique(st.session_state['y_test']) should provide these if y_test is label encoded.
                roc_fig = plot_roc_curve(st.session_state['y_test'], y_proba, labels)
                st.pyplot(roc_fig) # Display the captured figure

if __name__ == "__main__":
    main()
