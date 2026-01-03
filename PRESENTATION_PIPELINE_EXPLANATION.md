# ML-Pipeline und Modell - Präsentationserklärung

## 1. Datenbasis: Was bekommen wir?

### Herausforderungen der Rohdaten:

**Starkes Klassenungleichgewicht:**
- Klasse 0 (No Stenosis): ~15-20% der Daten
- Klasse 1 (Non-significant stenosis): ~25-30% der Daten  
- Klasse 2 (Significant Stenosis): ~50-60% der Daten
- **Problem:** Modell könnte zur Mehrheitsklasse neigen

**Hohe Dimensionalität, wenig Daten:**
- **634 numerische Features** + 7 kategorische Features
- **~1,700-2,300 Patienten** (nach Binary-Filterung)
- **Problem:** Curse of Dimensionality - viele Features, relativ wenige Samples
- **Lösung:** Feature Selection (SelectKBest mit mutual information)

**Weitere Datenprobleme:**
- Viele fehlende Werte (Missing at Random - MAR)
- Hohe Korrelation zwischen Features (>0.95)
- Abhängige Variablen (z.B. LDL, HDL, Cholesterin)

---

## 2. Pipeline: Wie wird daraus ein Modell?

### Schritt 1: Rohdaten → Preprocessing

**A) Datenbereinigung:**
- Entfernung abhängiger Variablen (LDL, HDL, Cholesterin, etc.)
- **Entfernung hochkorrelierter Features:**
  - Berechnung der **Pearson-Korrelationsmatrix** für alle numerischen Features
  - **Schwelle:** Korrelationskoeffizient > 0.95 (absoluter Wert)
  - **Methode:** 
    1. Extrahiere oberes Dreieck der Matrix (vermeidet Duplikate)
    2. Für jedes Feature: Finde alle anderen Features mit Korrelation > 0.95
    3. Entferne eines der beiden hochkorrelierten Features (behält das erste)
  - **Warum?** Hochkorrelierte Features enthalten redundante Information
  - **Beispiel:** Wenn Feature A und B eine Korrelation von 0.98 haben, wird B entfernt
- Ergebnis: ~200-220 Features (statt 634)

**B) Datenimputation (Missing at Random):**
- **Kategorische Features:** Most Frequent Imputation → OneHotEncoder
- **Numerische Features:** **kNN Imputation** (k=5) → **MinMax Scaling**
  - kNN nutzt ähnliche Patienten für bessere Imputation
  - MinMax Scaling: Werte auf [0,1] normalisiert (wie im Paper)

**C) Train/Test Split:**
- **75:25 Split** (wie im Paper)
- Stratified Split: Behält Klassenverteilung bei

### Schritt 2: Preprocessing → Training

**A) Feature Selection:**
- **SelectKBest** mit Mutual Information
- Wählt k=50-100 beste Features aus
- Reduziert Dimensionalität und Overfitting-Risiko

**B) SMOTE (Synthetic Minority Oversampling):**
- **Problem:** Klassenungleichgewicht
- **Lösung:** Erzeugt synthetische Samples für Minderheitsklassen
- k_neighbors=3 (reduziert Overfitting durch zu viele synthetische Daten)
- Wird nur auf Training-Set angewendet (nicht auf Test-Set!)

**C) XGBoost Classifier:**
- Gradient Boosting für Multiclass Classification
- **Hyperparameter (optimiert via Grid Search):**
  - `n_estimators`: 200 (Anzahl Bäume)
  - `learning_rate`: 0.02-0.03 (Lernrate)
  - `max_depth`: 4 (Baumtiefe - verhindert Overfitting)
  - `min_child_weight`: 3-5 (Regularisierung)
  - `subsample`: 0.8 (Row Sampling - 80% der Daten pro Baum)
  - `colsample_bytree`: 0.8 (Column Sampling - 80% der Features pro Baum)
  - `reg_lambda`: 2.0 (L2 Regularisierung)
  - `reg_alpha`: 1.0 (L1 Regularisierung)

### Schritt 3: Training → Validierung

**A) Cross-Validation:**
- **5-fold Stratified Cross-Validation** (wie im Paper)
- Scoring: Balanced Accuracy (berücksichtigt Klassenungleichgewicht)

**B) Evaluation Metriken:**
- **Overall Accuracy:** Gesamtgenauigkeit
- **Balanced Accuracy:** Durchschnittlicher Recall über alle Klassen
- **Per-Class Metriken:**
  - Sensitivity (Recall): Wie viele werden korrekt erkannt?
  - Precision: Wie viele der Vorhersagen sind korrekt?
  - F1 Score: Harmonisches Mittel aus Precision und Recall
- **Confusion Matrix:** Zeigt Verwechslungen zwischen Klassen

**C) Overfitting Check:**
- Vergleich Training vs. Test Performance
- Gap > 15% = Warnung vor Overfitting

### Schritt 4: Validierung → Erklärbarkeit

**A) Feature Importance:**
- XGBoost F-Score: Wie oft wird Feature in Bäumen verwendet?
- Top 10 Features zeigen wichtigste Prädiktoren

**B) SHAP-Analyse (Optional):**
- Erklärt einzelne Vorhersagen
- Zeigt Beitrag jedes Features zur Entscheidung

**C) ROC & PR Curves:**
- ROC: Trade-off zwischen True Positive Rate und False Positive Rate
- PR: Trade-off zwischen Precision und Recall
- AUC: Fläche unter der Kurve (höher = besser)

---

## 3. Modellentscheidung: Warum XGBoost?

### Warum XGBoost?

**1. Performance:**
- State-of-the-art für tabulare Daten
- Paper zeigt: XGBoost outperformed andere ML-Modelle

**2. Robustheit:**
- Kann mit fehlenden Werten umgehen (native Unterstützung)
- Robust gegenüber Outliers durch Gradient Boosting

**3. Regularisierung:**
- L1 (Lasso) und L2 (Ridge) Regularisierung
- Verhindert Overfitting bei hoher Dimensionalität

**4. Interpretierbarkeit:**
- Feature Importance verfügbar
- SHAP-Integration möglich

**5. Hyperparameter-Tuning:**
- Viele Parameter für Feinabstimmung
- Grid Search optimiert automatisch

### Warum diese Pipeline?

**kNN Imputation:**
- Nutzt ähnliche Patienten für bessere Imputation
- Besser als einfacher Median/Mode bei MAR-Daten

**MinMax Scaling:**
- Normalisiert Features auf [0,1]
- Wichtig für XGBoost und Feature Selection
- Wie im Paper verwendet

**SMOTE:**
- Adressiert Klassenungleichgewicht
- Erhöht Recall für Minderheitsklassen (besonders wichtig für Class 0)

**Feature Selection:**
- Reduziert Dimensionalität
- Fokus auf wichtigste Features
- Verhindert Overfitting

**75:25 Split:**
- Mehr Daten für Training (wichtig bei wenig Daten)
- Wie im Paper verwendet

---

## Zusammenfassung der Pipeline:

```
Rohdaten (634 Features, Klassenungleichgewicht)
    ↓
Datenbereinigung (Entfernung abhängiger/korrelierter Features)
    ↓
kNN Imputation + MinMax Scaling
    ↓
75:25 Train/Test Split
    ↓
Feature Selection (k=50-100)
    ↓
SMOTE (Oversampling)
    ↓
XGBoost Training (optimierte Hyperparameter)
    ↓
5-fold Cross-Validation
    ↓
Evaluation (Balanced Accuracy, Recall, Precision, F1)
    ↓
Feature Importance & SHAP
```

---

## Wichtige Punkte für die Präsentation:

1. **Herausforderungen betonen:** Klassenungleichgewicht, hohe Dimensionalität
2. **Jeden Schritt begründen:** Warum kNN? Warum SMOTE? Warum XGBoost?
3. **Paper-Referenz:** Methoden basieren auf ähnlicher Studie
4. **Evaluation:** Balanced Accuracy ist wichtig bei ungleichen Klassen
5. **Overfitting:** Ständige Überwachung durch Train/Test-Vergleich

