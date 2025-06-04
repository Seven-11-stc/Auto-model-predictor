import streamlit as st # Added for Streamlit compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
import plotly.express as px

def plot_metrics(results):
    """Create interactive plots for model comparison"""
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create metrics plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fig = px.bar(
        df, x='model', y=metrics,
        title='Model Performance Comparison',
        labels={'value': 'Score', 'variable': 'Metric'}
    )
    fig.update_layout(
        yaxis_range=[0, 1],
        height=600
    )
    st.plotly_chart(fig) # Changed for Streamlit

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix and return the figure object."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8)) # Create figure and axes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=labels, yticklabels=labels, ax=ax) # Use ax
    ax.set_xlabel('Predicted') # Use ax
    ax.set_ylabel('Actual') # Use ax
    ax.set_title('Confusion Matrix') # Use ax
    return fig # Return the figure object

def plot_roc_curve(y_true, y_pred_proba, labels):
    """Plot ROC curve and return the figure object."""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Ensure y_true is 1D array for roc_curve
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()

    for i in range(len(labels)):
        # Assuming labels correspond to 0, 1, ..., n_classes-1
        # If y_true contains original labels, they need to be mapped to 0..n-1 if not already
        y_true_class = (y_true == labels[i]).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_class, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fig, ax = plt.subplots(figsize=(10, 8)) # Create figure and axes
    for i in range(len(labels)):
        ax.plot(fpr[i], tpr[i], # Use ax
                label=f'Class {labels[i]} (area = {roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--') # Use ax
    ax.set_xlim([0.0, 1.0]) # Use ax
    ax.set_ylim([0.0, 1.05]) # Use ax
    ax.set_xlabel('False Positive Rate') # Use ax
    ax.set_ylabel('True Positive Rate') # Use ax
    ax.set_title('Receiver Operating Characteristic') # Use ax
    ax.legend(loc="lower right") # Use ax
    return fig # Return the figure object
