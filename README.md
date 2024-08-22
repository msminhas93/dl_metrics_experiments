## Evaluating Classifier Performance with Imbalanced Data

- [Evaluating Classifier Performance with Imbalanced Data](#evaluating-classifier-performance-with-imbalanced-data)
  - [Introduction](#introduction)
  - [Definitions of Metrics](#definitions-of-metrics)
    - [Confusion Matrix](#confusion-matrix)
    - [Accuracy](#accuracy)
    - [Precision](#precision)
    - [Recall (Sensitivity)](#recall-sensitivity)
    - [F1-Score](#f1-score)
    - [AUROC (Area Under the ROC Curve)](#auroc-area-under-the-roc-curve)
  - [Confusion Matrices Analysis](#confusion-matrices-analysis)
  - [ROC Curves Analysis](#roc-curves-analysis)
  - [Precision, Recall, and F1-Score Analysis](#precision-recall-and-f1-score-analysis)
  - [Implications and Recommendations](#implications-and-recommendations)
  - [Conclusion](#conclusion)


### Introduction

In machine learning, evaluating classifiers involves understanding how well models predict outcomes using metrics like accuracy, precision, recall, F1-score, and AUROC. Class imbalance can significantly affect these metrics, leading to misleading interpretations. This report analyzes the impact of class imbalance and different classifier strategies on these metrics, using confusion matrices and ROC curves for visualization.

### Definitions of Metrics

#### Confusion Matrix

A confusion matrix is a table used to evaluate the performance of a classification model. It provides a breakdown of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

**Confusion Matrix Layout:**

|          | Predicted Negative | Predicted Positive |
|----------|--------------------|--------------------|
| Actual Negative | TN                 | FP                 |
| Actual Positive | FN                 | TP                 |

#### Accuracy

Accuracy measures the ratio of correctly predicted samples to the total samples.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### Precision

Precision measures the ratio of true positive predictions to the total positive predictions.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

#### Recall (Sensitivity)

Recall measures the ratio of true positive predictions to the actual positive instances.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

#### F1-Score

The F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics.

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

#### AUROC (Area Under the ROC Curve)

AUROC measures the model's ability to distinguish between classes across different thresholds. It is the area under the ROC curve, that plots the true positive rate (TPR) against the false positive rate (FPR). A higher AUROC indicates better performance, with 0.5 representing random guessing and 1.0 indicating perfect discrimination.

### Confusion Matrices Analysis

Based on the provided confusion matrices:

1. **Probability = 0 (All-Zero Classifier):**
   - **Imbalance 0.5:** High TN, zero TP, FP, and FN.
   - **Imbalance 0.1:** High TN, zero TP, indicating poor performance on minority class.
   - **Imbalance 0.9:** High TN, zero TP, similar to imbalance 0.1.

2. **Probability = 0.5 (Random Classifier):**
   - **Imbalance 0.5:** Mixed TP, TN, FP, and FN.
   - **Imbalance 0.1:** Skewed towards the majority class, resulting in more TN.
   - **Imbalance 0.9:** Skewed towards the minority class, with more TP.

3. **Probability = 1 (All-One Classifier):**
   - **Imbalance 0.5:** High TP, zero TN, FP, and FN.
   - **Imbalance 0.1:** High TP, zero TN, indicating poor performance on majority class.
   - **Imbalance 0.9:** High TP, zero TN, similar to imbalance 0.1.

### ROC Curves Analysis

The ROC curves show the trade-off between true positive rate (TPR) and false positive rate (FPR):

- All curves are close to the diagonal, indicating poor discrimination ability across scenarios.
- Random classifiers (p=0.5) show slight variations but generally align with the diagonal, confirming random guessing.

### Precision, Recall, and F1-Score Analysis

- **Precision:** High for all-zero classifiers when predicting the majority class, but zero for minority class predictions.
- **Recall:** High for all-one classifiers when predicting the minority class, but zero for majority class predictions.
- **F1-Score:** Balances precision and recall, but can be misleading if one is significantly lower.

### Implications and Recommendations

- **Accuracy**: Can be misleading in imbalanced datasets. High accuracy might not reflect true performance.
- **Precision and Recall**: Use these to understand trade-offs. High precision with low recall indicates many false negatives, while high recall with low precision indicates many false positives.
- **AUROC**: Useful for comparing models across thresholds. Less affected by imbalance but still needs careful interpretation.
- **Confusion Matrices**: Provide detailed insights into classifier performance. Use them to identify specific areas of improvement.

### Conclusion

Understanding the impact of class imbalance and classifier behavior is crucial for accurate model evaluation. Metrics should be interpreted contextually, considering class distribution and the specific use case. Combining multiple metrics and visualizations like ROC curves and confusion matrices can provide a more comprehensive assessment of model performance.