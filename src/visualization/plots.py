import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_style():
    """Set a premium dark style for plots."""
    plt.style.use('dark_background')
    sns.set_palette("viridis")

def plot_distribution(data, column, title, save_path=None):
    """Plot the distribution of a numeric column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, color='#4CAF50')
    plt.title(title, fontsize=15)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_matrix(df, title="Correlation Matrix", save_path=None):
    """Plot a correlation heatmap for numeric features."""
    plt.figure(figsize=(15, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='magma', center=0)
    plt.title(title, fontsize=15)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """Plot a scatter plot of actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='#9b59b6')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(title, fontsize=15)
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_model_comparison(results_dict, metric_name="RMSE", save_path=None):
    """Compare different models based on a metric."""
    plt.figure(figsize=(12, 6))
    models = list(results_dict.keys())
    values = list(results_dict.values())
    sns.barplot(x=models, y=values, palette="rocket")
    plt.title(f"Model Comparison ({metric_name})", fontsize=15)
    plt.ylabel(metric_name)
    if save_path:
        plt.savefig(save_path)
    plt.show()
