"""
evaluate.py
-----------
Evaluation utilities: accuracy table, ROC curves,
confusion matrices, and SHAP feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_curve, auc, confusion_matrix,
)


def evaluate_all(trained_models: dict, X_test, y_test) -> dict:
    """
    Run predictions for all models and return a results dict.

    Returns
    -------
    results : {model_name: {accuracy, report, y_pred, y_proba, cm}}
    """
    results = {}
    for name, model in trained_models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report':   classification_report(y_test, y_pred, output_dict=True),
            'y_pred':   y_pred,
            'y_proba':  y_proba,
            'cm':       confusion_matrix(y_test, y_pred),
        }
    return results


def summary_table(results: dict) -> pd.DataFrame:
    """Return a DataFrame comparing accuracy, precision, recall, F1."""
    rows = []
    for name, res in results.items():
        wa = res['report']['weighted avg']
        rows.append({
            'Model':     name,
            'Accuracy':  round(res['accuracy'], 3),
            'Precision': round(wa['precision'], 3),
            'Recall':    round(wa['recall'], 3),
            'F1-Score':  round(wa['f1-score'], 3),
        })
    return pd.DataFrame(rows).sort_values('Accuracy', ascending=False)


def plot_roc(results: dict, y_test, save_path: str = None):
    """Plot ROC curves for all models."""
    colors = ['#1565C0', '#E65100', '#2E7D32', '#7B1FA2']
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Model Comparison', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_confusion_matrices(results: dict, save_path: str = None):
    """Plot confusion matrices for all models in a single figure."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))

    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(
            res['cm'], annot=True, fmt='d', cmap='Greens', ax=ax,
            cbar=False,
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'],
        )
        ax.set_title(name, fontweight='bold', fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_feature_importance(rf_model, feature_names: list, save_path: str = None):
    """Bar chart of Random Forest feature importances."""
    importances = pd.Series(
        rf_model.feature_importances_, index=feature_names
    ).sort_values(ascending=True)

    colors = ['#2E7D32' if v == importances.max() else '#90CAF9'
              for v in importances.values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importances.index, importances.values, color=colors, edgecolor='white')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance — Random Forest', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
