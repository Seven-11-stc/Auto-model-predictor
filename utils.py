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
    fig.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, labels):
    """Plot ROC curve"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    for i in range(len(labels)):
        plt.plot(fpr[i], tpr[i],
                label=f'Class {labels[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
