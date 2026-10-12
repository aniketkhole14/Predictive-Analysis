# Breast Cancer Diagnosis Prediction using Binomial Logistic Regression

**Course:** Predictive Analytics

## 1. Problem Statement
Early and accurate classification of a breast tumour as **benign** or **malignant** helps clinicians decide on treatment quickly. This project builds a **binomial logistic regression** model that predicts whether a tumour is malignant from measurements of cell nuclei taken from a fine needle aspirate (FNA) image.

## 2. Dataset
- **Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository
- Link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- 569 samples, 30 numeric features (mean, standard error and "worst" value of 10 nuclear characteristics such as radius, texture, concavity, symmetry)
- Target: malignant (212) vs benign (357); no missing values
- The same data ships with scikit-learn (`sklearn.datasets.load_breast_cancer`), so the code runs without downloading anything.

## 3. Method
1. Exploratory data analysis (class balance, correlations, boxplots)
2. Stratified 80/20 train/test split (455 train, 114 test)
3. Pipeline: `StandardScaler` followed by `LogisticRegression` (L2 penalty)
4. Regularisation strength `C` tuned with 5-fold stratified `GridSearchCV` (scoring = ROC-AUC); best `C = 1`
5. Evaluation on the held-out test set
6. Interpretation through coefficients and odds ratios

Binomial logistic regression models the log-odds of the event:

`log( p / (1 - p) ) = b0 + b1*x1 + ... + bk*xk`, where `p = P(malignant)`.

## 4. Results (test set)
| Metric | Value |
|---|---|
| Accuracy | 0.965 |
| Precision (malignant) | 0.975 |
| Recall (malignant) | 0.929 |
| F1 | 0.951 |
| ROC-AUC | 0.996 |
| 5-fold CV accuracy (train) | 0.974 +/- 0.015 |

Top predictors (odds ratio of malignancy per +1 SD increase): worst texture (4.20), radius error (3.43), worst symmetry (2.89), mean concave points (2.59), worst concavity (2.49).

## 5. Discussion
- The model separates the two classes very well (AUC about 0.996) while staying fully interpretable.
- Recall for malignant cases (0.93) is the weaker side; in a medical setting a lower decision threshold could be used to reduce missed cancers at the cost of more false alarms.
- Many features are highly correlated (radius, perimeter, area), so individual coefficients should be read with care; L2 regularisation keeps them stable.
- Limitations: small dataset, a single split, no external validation.

## 6. Project Structure
```
train.py            # full pipeline
requirements.txt
outputs/
  metrics.json
  summary_statistics.csv
  coefficients_odds_ratios.csv
  logistic_regression_model.joblib
  01_class_distribution.png ... 06_feature_importance.png
```

## 7. How to Run
```bash
pip install -r requirements.txt
python train.py
```
