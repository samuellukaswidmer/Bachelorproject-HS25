"""
Model Comparison Visualization - Binary Models
Erstellt Vergleichsgrafiken für Binary-Modelle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def get_binary_performances():
    """
    Hardcodierte Performance-Werte für Binary-Modelle
    Bitte aktualisieren Sie diese Werte, falls sie falsch sind.
    """
    performances = {
        'XGBoost': {
            'accuracy': 0.8100,
            'balanced_accuracy': 0.6829,
            'roc_auc': 0.8500,
            'recall_class1': 0.8746,
            'f1_score_class1': 0.8800
        },
        'Random Forest': {
            'accuracy': 0.7946,
            'balanced_accuracy': 0.7055,
            'roc_auc': 0.7867,
            'recall_class1': 0.8631,
            'f1_score_class1': 0.8681
        },
        'TabNet': {
            'accuracy': 0.7708,
            'balanced_accuracy': 0.6458,
            'roc_auc': 0.8200,
            'recall_class1': 0.8669,
            'f1_score_class1': 0.8555
        },
        'Logistic Regression': {
            'accuracy': 0.7000,
            'balanced_accuracy': 0.6991,
            'roc_auc': 0.7500,
            'recall_class1': 0.6996,
            'f1_score_class1': 0.7800
        },
        'SVM': {
            'accuracy': 0.7300,
            'balanced_accuracy': 0.6934,
            'roc_auc': 0.7600,
            'recall_class1': 0.7567,
            'f1_score_class1': 0.8100
        }
    }
    return performances

def plot_binary_comparison(figsize=(16, 10)):
    """
    Erstellt eine umfassende Vergleichsvisualisierung für Binary-Modelle
    basierend auf: Accuracy, Balanced Accuracy, ROC AUC, Recall Class 1, F1 Score Class 1
    """
    performances = get_binary_performances()
    
    df = pd.DataFrame(performances).T
    
    metrics = ['accuracy', 'balanced_accuracy', 'roc_auc', 'recall_class1', 'f1_score_class1']
    metric_labels = ['Accuracy', 'Balanced Accuracy', 'ROC AUC', 'Recall Class 1', 'F1 Score Class 1']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df.index)))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        bars = ax.bar(range(len(df)), df[metric], color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        for i, (bar, val) in enumerate(zip(bars, df[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=15, ha='right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0, max(df[metric].max() * 1.2, 0.8)])
        ax.axhline(y=0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Baseline (0.5)')
        if idx == 0:
            ax.legend(fontsize=9)
    
    axes[5].axis('off')
    
    plt.suptitle('Binary Model Performance Comparison\nAccuracy, Balanced Accuracy, ROC AUC, Recall Class 1, F1 Score Class 1', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig, axes

def plot_binary_heatmap(figsize=(12, 8)):
    """
    Erstellt eine Heatmap für alle Metriken aller Binary-Modelle
    """
    performances = get_binary_performances()
    df = pd.DataFrame(performances).T
    
    metrics = ['accuracy', 'balanced_accuracy', 'roc_auc', 'recall_class1', 'f1_score_class1']
    metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'ROC AUC', 'Recall\nClass 1', 'F1 Score\nClass 1']
    
    df_subset = df[metrics]
    df_subset.columns = metric_labels
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(df_subset, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, ax=ax,
                linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
    
    ax.set_title('Binary Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
    ax.set_ylabel('Models', fontsize=13, fontweight='bold')
    
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig, ax

def plot_binary_radar(figsize=(12, 10)):
    """
    Erstellt ein Radar Chart für alle Metriken aller Binary-Modelle
    """
    performances = get_binary_performances()
    df = pd.DataFrame(performances).T
    
    metrics = ['accuracy', 'balanced_accuracy', 'roc_auc', 'recall_class1', 'f1_score_class1']
    metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'ROC AUC', 'Recall\nClass 1', 'F1 Score\nClass 1']
    
    df_subset = df[metrics]
    
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_subset.index)))
    
    for idx, (model_name, row) in enumerate(df_subset.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.set_title('Binary Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    return fig, ax

def plot_binary_grouped_bars(figsize=(14, 8)):
    """
    Erstellt gruppierte Bar Charts für alle Metriken
    """
    performances = get_binary_performances()
    df = pd.DataFrame(performances).T
    
    metrics = ['accuracy', 'balanced_accuracy', 'roc_auc', 'recall_class1', 'f1_score_class1']
    metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'ROC AUC', 'Recall\nClass 1', 'F1 Score\nClass 1']
    
    x = np.arange(len(df.index))
    width = 0.15
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for metric, label, color in zip(metrics, metric_labels, colors):
        offset = width * multiplier
        bars = ax.bar(x + offset, df[metric], width, label=label, alpha=0.8, color=color, edgecolor='black', linewidth=1)
        
        for bar, val in zip(bars, df[metric]):
            height = bar.get_height()
            if height > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        multiplier += 1
    
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Binary Model Performance Comparison - All Metrics', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(df.index, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    return fig, ax

if __name__ == "__main__":
    print("=" * 80)
    print("Binary Model Comparison Visualization")
    print("=" * 80)
    
    performances = get_binary_performances()
    print("\nHardcodierte Performance-Werte:")
    print("=" * 80)
    df = pd.DataFrame(performances).T
    print(df.to_string())
    print("\n" + "=" * 80)
    print("Hinweis: Falls diese Werte falsch sind, bitte in der Funktion")
    print("         'get_binary_performances()' anpassen!")
    print("=" * 80)
    
    print("\n1. Creating comprehensive comparison (5 metrics)...")
    fig, axes = plot_binary_comparison()
    plt.savefig('binary_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: binary_comparison_comprehensive.png")
    plt.close()
    
    print("\n2. Creating heatmap...")
    fig, ax = plot_binary_heatmap()
    plt.savefig('binary_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: binary_comparison_heatmap.png")
    plt.close()
    
    print("\n3. Creating radar chart...")
    fig, ax = plot_binary_radar()
    plt.savefig('binary_comparison_radar.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: binary_comparison_radar.png")
    plt.close()
    
    print("\n4. Creating grouped bar chart...")
    fig, ax = plot_binary_grouped_bars()
    plt.savefig('binary_comparison_grouped_bars.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: binary_comparison_grouped_bars.png")
    plt.close()
    
    print("\n5. Saving comparison table...")
    df.to_csv('binary_comparison_table.csv')
    print("   ✓ Saved: binary_comparison_table.csv")
    
    print("\n" + "=" * 80)
    print("All visualizations created successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • binary_comparison_comprehensive.png")
    print("  • binary_comparison_heatmap.png")
    print("  • binary_comparison_radar.png")
    print("  • binary_comparison_grouped_bars.png")
    print("  • binary_comparison_table.csv")
    print("=" * 80)

