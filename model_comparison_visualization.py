"""
Model Comparison Visualization
Erstellt Vergleichsgrafiken für alle Modelle im Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_comparison_table(models_data):
    """
    Erstellt eine Vergleichstabelle für alle Modelle
    
    models_data: dict mit folgender Struktur:
    {
        'Model Name': {
            'accuracy': float,
            'balanced_accuracy': float,
            'f1_score': float,
            'roc_auc': float,
            'average_precision': float,
            'recall_class2': float,  # für Multiclass
            'train_time': float  # optional
        }
    }
    """
    df = pd.DataFrame(models_data).T
    return df

def plot_roc_comparison(roc_data, figsize=(10, 8)):
    """
    Plottet ROC Curves für alle Modelle
    
    roc_data: dict mit {'Model Name': (fpr, tpr, auc)}
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig, ax

def plot_pr_comparison(pr_data, figsize=(10, 8)):
    """
    Plottet Precision-Recall Curves für alle Modelle
    
    pr_data: dict mit {'Model Name': (recall, precision, ap)}
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (recall, precision, ap) in pr_data.items():
        ax.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig, ax

def plot_metrics_bar_chart(models_data, metrics=['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc'], 
                          figsize=(14, 6)):
    """
    Erstellt Bar Chart für verschiedene Metriken
    
    models_data: dict wie in create_comparison_table
    metrics: Liste der Metriken die geplottet werden sollen
    """
    df = pd.DataFrame(models_data).T
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=True)
    
    if len(metrics) == 1:
        axes = [axes]
    
    x = np.arange(len(df.index))
    width = 0.6
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(df.index)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(x, df[metric], width, label=metric, color=colors)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, df[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig, axes

def plot_metrics_heatmap(models_data, metrics=['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc', 'average_precision'],
                        figsize=(10, 6)):
    """
    Erstellt Heatmap für Metriken-Vergleich
    """
    df = pd.DataFrame(models_data).T
    df_subset = df[metrics]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize to 0-1 for better visualization
    df_normalized = df_subset.copy()
    
    sns.heatmap(df_normalized, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score (normalized)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Models', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig, ax

def plot_radar_chart(models_data, metrics=['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc', 'average_precision'],
                    figsize=(10, 10)):
    """
    Erstellt Radar Chart für Multi-Metriken-Vergleich
    """
    df = pd.DataFrame(models_data).T
    df_subset = df[metrics]
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(df_subset.index)))
    
    for idx, (model_name, row) in enumerate(df_subset.iterrows()):
        values = row.values.tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    return fig, ax

def plot_comprehensive_comparison(models_data, roc_data, pr_data, figsize=(16, 12)):
    """
    Erstellt eine umfassende Vergleichsgrafik mit mehreren Subplots
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. ROC Curves (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        ax1.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    for model_name, (recall, precision, ap) in pr_data.items():
        ax2.plot(recall, precision, label=f'{model_name} (AP={ap:.3f})', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Bar Chart - Balanced Accuracy (middle left)
    df = pd.DataFrame(models_data).T
    ax3 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.Set3(np.linspace(0, 1, len(df.index)))
    bars = ax3.bar(range(len(df)), df['balanced_accuracy'], color=colors)
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df.index, rotation=45, ha='right')
    ax3.set_ylabel('Balanced Accuracy')
    ax3.set_title('Balanced Accuracy Comparison', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, df['balanced_accuracy'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Bar Chart - F1 Score (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.bar(range(len(df)), df['f1_score'], color=colors)
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df.index, rotation=45, ha='right')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score Comparison', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, df['f1_score'])):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Heatmap (bottom, spanning both columns)
    ax5 = fig.add_subplot(gs[2, :])
    metrics_subset = ['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc', 'average_precision']
    df_heatmap = df[metrics_subset]
    sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax5,
                cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='gray')
    ax5.set_title('Performance Metrics Heatmap', fontweight='bold', pad=10)
    ax5.set_xlabel('Metrics', fontsize=11)
    ax5.set_ylabel('Models', fontsize=11)
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold', y=0.995)
    return fig

# Example usage (you would replace this with your actual data):
if __name__ == "__main__":
    # Example data structure - replace with your actual results
    example_models_data = {
        'Logistic Regression': {
            'accuracy': 0.643,
            'balanced_accuracy': 0.636,
            'f1_score': 0.687,
            'roc_auc': 0.701,
            'average_precision': 0.751
        },
        'SVM (Linear)': {
            'accuracy': 0.650,
            'balanced_accuracy': 0.640,
            'f1_score': 0.690,
            'roc_auc': 0.710,
            'average_precision': 0.760
        },
        'Random Forest': {
            'accuracy': 0.652,
            'balanced_accuracy': 0.635,
            'f1_score': 0.712,
            'roc_auc': 0.710,
            'average_precision': 0.771
        },
        'XGBoost Binary': {
            'accuracy': 0.669,
            'balanced_accuracy': 0.648,
            'f1_score': 0.733,
            'roc_auc': 0.720,
            'average_precision': 0.755
        },
        'XGBoost Multiclass': {
            'accuracy': 0.539,
            'balanced_accuracy': 0.490,
            'f1_score': 0.510,
            'roc_auc': 0.680,
            'average_precision': 0.720
        }
    }
    
    # Example ROC data (fpr, tpr, auc)
    # You would extract these from your model evaluations
    example_roc_data = {
        'Logistic Regression': (np.linspace(0, 1, 100), np.linspace(0, 1, 100)**0.8, 0.701),
        'SVM (Linear)': (np.linspace(0, 1, 100), np.linspace(0, 1, 100)**0.75, 0.710),
        'Random Forest': (np.linspace(0, 1, 100), np.linspace(0, 1, 100)**0.7, 0.710),
        'XGBoost Binary': (np.linspace(0, 1, 100), np.linspace(0, 1, 100)**0.65, 0.720),
    }
    
    # Example PR data (recall, precision, ap)
    example_pr_data = {
        'Logistic Regression': (np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)**0.5, 0.751),
        'SVM (Linear)': (np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)**0.45, 0.760),
        'Random Forest': (np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)**0.4, 0.771),
        'XGBoost Binary': (np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)**0.35, 0.755),
    }
    
    # Create visualizations
    print("Creating comparison visualizations...")
    
    # 1. Comprehensive comparison
    fig = plot_comprehensive_comparison(example_models_data, example_roc_data, example_pr_data)
    plt.savefig('model_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison_comprehensive.png")
    
    # 2. Individual plots
    fig, ax = plot_roc_comparison(example_roc_data)
    plt.savefig('model_comparison_roc.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison_roc.png")
    
    fig, ax = plot_pr_comparison(example_pr_data)
    plt.savefig('model_comparison_pr.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison_pr.png")
    
    fig, axes = plot_metrics_bar_chart(example_models_data)
    plt.savefig('model_comparison_bars.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison_bars.png")
    
    fig, ax = plot_metrics_heatmap(example_models_data)
    plt.savefig('model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison_heatmap.png")
    
    fig, ax = plot_radar_chart(example_models_data)
    plt.savefig('model_comparison_radar.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison_radar.png")
    
    print("\nAll visualizations created successfully!")

