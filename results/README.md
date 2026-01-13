# Model Comparison Visualization

This directory contains the model comparison visualization script and all generated comparison plots.

## Usage

Run the visualization script from this directory:

```bash
cd results
python model_comparison_visualization.py
```

## Generated Files

The script generates the following files:

- `model_comparison_comprehensive.png` - Comprehensive comparison with multiple subplots
- `model_comparison_roc.png` - ROC curves for all models
- `model_comparison_pr.png` - Precision-Recall curves for all models
- `model_comparison_bars.png` - Bar charts for key metrics
- `model_comparison_heatmap.png` - Heatmap of all metrics
- `model_comparison_radar.png` - Radar chart for multi-metric comparison
- `model_comparison_multiclass_metrics.png` - Detailed multiclass metrics comparison
- `model_comparison_table.csv` - CSV table with all metrics

## Models Included

The script automatically loads metrics from:

**Binary Classification:**
- XGBoost Binary
- Random Forest Binary
- Logistic Regression Binary
- SVM Binary
- TabNet Binary

**Multiclass Classification:**
- XGBoost Multiclass
- Random Forest Multiclass
- Logistic Regression Multiclass
- SVM Multiclass
- TabNet Multiclass

## Notes

- The script looks for notebooks in `../binary-classification/` and `../multiclass-classification/`
- If metrics cannot be extracted from notebooks, default values are used
- All output files are saved in this directory

