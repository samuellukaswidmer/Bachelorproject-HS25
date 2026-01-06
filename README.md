# Bachelorproject-HS25

Machine Learning Classification Models for Coronary Artery Stenosis Detection

## Project Structure

```
Bachelorproject-HS25/
├── README.md
├── requirements.txt
├── data/
│   └── T49.2_Sep2025_1_StGallen.csv          # Dataset
├── binary-classification/                     # Binary classification models
│   ├── xgboost-binary-classification-final.ipynb
│   ├── random-forest-binary-classification-final.ipynb
│   ├── logistic-regression-binary-classification-final.ipynb
│   ├── support-vector-machine-binary-classification-final.ipynb
│   └── tabnet-binary-classification-final.ipynb
├── multiclass-classification/                # Multiclass classification models
│   ├── xgboost-multiclass-classification-final.ipynb
│   ├── random-forest-multiclass-classification-final.ipynb
│   ├── logistic-regression-multiclass-classification-final.ipynb
│   ├── support-vector-machine-multiclass-classification-final.ipynb
│   └── tabnet-multiclass-classification-final.ipynb
├── archive/                                  # Experimental and old notebooks
│   ├── stenosis.ipynb
│   ├── support_vector_machine.ipynb
│   ├── logistic_regression.ipynb
│   ├── xgboost-experiments.ipynb
│   ├── xgboost-binary-clean-experiments.ipynb
│   └── xgboost-multiclass-classification-experiments.ipynb
└── results/                                  # Model comparison visualizations
    ├── model_comparison_visualization.py
    ├── model_comparison_table.csv
    └── model_comparison_*.png                 # Generated comparison plots
```

## Models

### Binary Classification
- **XGBoost**: Gradient boosting classifier
- **Random Forest**: Ensemble tree-based classifier
- **Logistic Regression**: Linear classifier
- **Support Vector Machine (SVM)**: RBF kernel SVM
- **TabNet**: Deep learning tabular model

### Multiclass Classification
- **XGBoost**: Gradient boosting classifier (3 classes)
- **Random Forest**: Ensemble tree-based classifier (3 classes)
- **Logistic Regression**: Linear classifier (3 classes)
- **Support Vector Machine (SVM)**: RBF kernel SVM (3 classes)
- **TabNet**: Deep learning tabular model (3 classes)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run notebooks from their respective directories:
   - Binary classification notebooks: `binary-classification/`
   - Multiclass classification notebooks: `multiclass-classification/`

3. Generate model comparison visualizations:
```bash
cd results
python model_comparison_visualization.py
```

## Dataset

The dataset (`T49.2_Sep2025_1_StGallen.csv`) contains medical features for coronary artery stenosis classification:
- **Binary target**: 0 = No Stenosis, 1 = Significant Stenosis
- **Multiclass target**: 0 = No Stenosis, 1 = Non-significant stenosis, 2 = Significant Stenosis

## Notes

- All notebooks use relative paths: `../data/T49.2_Sep2025_1_StGallen.csv`
- Model comparison script should be run from the `results/` directory
- Experimental notebooks are archived in the `archive/` folder
