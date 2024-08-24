## Evaluating Classifier Performance with Imbalanced Data

- [Evaluating Classifier Performance with Imbalanced Data](#evaluating-classifier-performance-with-imbalanced-data)
  - [Introduction](#introduction)
  - [Definitions of Metrics](#definitions-of-metrics)
    - [Threshold dependent metrics](#threshold-dependent-metrics)
      - [Confusion Matrix](#confusion-matrix)
        - [Structure](#structure)
        - [Uses](#uses)
        - [Importance](#importance)
      - [Accuracy](#accuracy)
      - [Precision](#precision)
      - [Recall (Sensitivity)](#recall-sensitivity)
      - [F1-Score](#f1-score)
    - [Threshold independent metrics](#threshold-independent-metrics)
      - [AUROC (Area Under the ROC Curve)](#auroc-area-under-the-roc-curve)
      - [PR Curve](#pr-curve)
  - [Experiment Setup](#experiment-setup)
  - [PR Curve Analysis for Different Classifier Types and Class Imbalance](#pr-curve-analysis-for-different-classifier-types-and-class-imbalance)
  - [Classifier Types](#classifier-types)
  - [Impact of Class Imbalance](#impact-of-class-imbalance)
  - [Conclusion](#conclusion)
  - [AUROC Analysis for Different Classifier Types and Class Imbalance](#auroc-analysis-for-different-classifier-types-and-class-imbalance)
    - [Classifier Types](#classifier-types-1)
    - [Impact of Class Imbalance](#impact-of-class-imbalance-1)
  - [Conclusion](#conclusion-1)
  - [Accuracy Analysis for Different Classifier Types and Class Imbalance](#accuracy-analysis-for-different-classifier-types-and-class-imbalance)
    - [Classifier Types](#classifier-types-2)
    - [Impact of Class Imbalance](#impact-of-class-imbalance-2)
  - [Conclusion](#conclusion-2)
  - [Precision Analysis for Different Classifier Types and Class Imbalance](#precision-analysis-for-different-classifier-types-and-class-imbalance)
    - [Classifier Types](#classifier-types-3)
    - [Impact of Class Imbalance](#impact-of-class-imbalance-3)
  - [Conclusion](#conclusion-3)
  - [Recall Analysis for Different Classifier Types and Class Imbalance](#recall-analysis-for-different-classifier-types-and-class-imbalance)
    - [Classifier Types](#classifier-types-4)
    - [Impact of Class Imbalance](#impact-of-class-imbalance-4)
  - [Conclusion](#conclusion-4)
  - [F1 Score Analysis for Different Classifier Types and Class Imbalance](#f1-score-analysis-for-different-classifier-types-and-class-imbalance)
    - [Classifier Types](#classifier-types-5)
    - [Impact of Class Imbalance](#impact-of-class-imbalance-5)
  - [Conclusion](#conclusion-5)
  - [Confusion Matrix Analysis for Different Classifier Types and Class Imbalance](#confusion-matrix-analysis-for-different-classifier-types-and-class-imbalance)
    - [Classifier Types](#classifier-types-6)
    - [Impact of Class Imbalance](#impact-of-class-imbalance-6)
  - [Conclusion](#conclusion-6)
  - [Confusion Matrices Analysis](#confusion-matrices-analysis)
  - [ROC Curves Analysis](#roc-curves-analysis)
  - [Precision, Recall, and F1-Score Analysis](#precision-recall-and-f1-score-analysis)
  - [Implications and Recommendations](#implications-and-recommendations)
  - [Conclusion](#conclusion-7)
  - [Key Points](#key-points)
  - [References](#references)


### Introduction

If you are into deep learning or machine learning you would have wanted to understand the performance of your model. There are several metrics available for measuring performance. However, if you get stuck in the horrible situation of solely focusing numbers without looking closely at what is happening, you are just deluding yourself and not assessing the situation fairly.

Using this post I want to spread awareness about some common pitfalls that you can encounter and how to potentially avoid them. Hopefully it will add value to your skills. I will be limiting the scope of the discussion to binary classification metrics which is probably one of the most common modeling problem. 

Evaluating a classifier involves understanding how well your model makes prediction given certain input(s) using metrics like accuracy, precision, recall, F1-score, and AUROC. There is no magic metric and value that can be applicable to every use case. These will depend on several factors including your data distribution, tolerance to false positives of false negatives etc. 

Class imbalance can significantly affect the metrics, leading to misleading interpretations. We will analyze the impact of class imbalance and different classifiers namely: random, always 0 and always 1, on these metrics. Let us start with the definition of these metics. 

### Definitions of Metrics
#### Threshold dependent metrics
All the metrics in this section are dependent on the choice of threshold and will change based on the choice. So always ask what was the threshold used for calculating them. 

##### Confusion Matrix

A **confusion matrix** is a table used to evaluate the performance of a classification algorithm. It provides a detailed breakdown of the model's predictions compared to the actual outcomes. The matrix consists of four key components:

- **True Positives (TP)**: The number of instances correctly predicted as positive.
- **True Negatives (TN)**: The number of instances correctly predicted as negative.
- **False Positives (FP)**: The number of instances incorrectly predicted as positive (also known as Type I error).
- **False Negatives (FN)**: The number of instances incorrectly predicted as negative (also known as Type II error).

###### Structure

The confusion matrix is typically structured as follows:

|                | **Predicted Positive** | **Predicted Negative** |
|----------------|------------------------|------------------------|
| **Actual Positive** | True Positives (TP)      | False Negatives (FN)     |
| **Actual Negative** | False Positives (FP)     | True Negatives (TN)      |

###### Uses

- **Performance Metrics**: From the confusion matrix, you can derive various performance metrics such as accuracy, precision, recall, and F1-score.
- **Error Analysis**: It helps identify specific areas where the model is making errors, allowing for targeted improvements.

###### Importance

The confusion matrix provides a comprehensive view of how well a classification model is performing, especially in terms of distinguishing between different classes. It is particularly useful in imbalanced datasets, where accuracy alone can be misleading.

##### Accuracy

Accuracy measures the ratio of correctly predicted samples to the total samples. This is probably the most misleading and useless metric you could blindly rely on. Strongly susceptible to class imbalance and gives near perfect scores for extremely imbalanced datasets. No real dataset will be balanced unless you make it so. "For heavily imbalanced datasets, where one class appears very rarely, say 1% of the time, a model that predicts negative 100% of the time would score 99% on accuracy, despite being useless." [3] The range of accuracy is 0 to 1. 

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

##### Precision

Precision measures the ratio of true positive predictions to the total positive predictions. From an information retrieval perspective, it measures the faction of relevant instances among the retrieved instances [2]. Its range is 0 to 1. This means out of all the positive predictions of the model how many of those were correct. An example where precision would be important is spam filter, you wouldn't want your important emails to be misclassified as spam. 

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

##### Recall (Sensitivity)

Recall measures the ratio of true positive predictions to the actual positive instances. It is the fraction of relevant instances there were retrieved. In order words, out of all the positives in your set, how many were correctly identified as positive by your model. The range of recall is 0 to 1. Precision and recall are more robust to class imbalance. A typical example where recall is important is in the case of cancer detection. A false negative is several times worse than a false positive. Precision and recall are often competing metrics that have inverse relationship and so you'd typically value one over the other. [3]

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

##### F1-Score

The F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics. If you value for precision and recall, you can look at the F1-score. 

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

#### Threshold independent metrics

##### AUROC (Area Under the ROC Curve)
The **Area Under the Receiver Operating Characteristic (AUROC)** curve is a performance metric used to evaluate the ability of a binary classifier to distinguish between positive and negative classes across all possible classification thresholds. Here's a detailed definition:

- **ROC Curve**: The ROC curve is a graphical plot that illustrates the performance of a binary classifier by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

- **AUC (Area Under the Curve)**: The AUROC is the area under the ROC curve. It provides a single scalar value that summarizes the overall performance of the classifier. The AUROC ranges from 0 to 1:
  - **0.5**: Indicates no discriminative ability, equivalent to random guessing.
  - **1.0**: Represents perfect discrimination, where the model perfectly distinguishes between classes.

- **Interpretation**: The AUROC can be interpreted as the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance by the classifier.

- **Threshold-Invariant**: It evaluates performance across all classification thresholds.

- **Use Cases**: AUROC is particularly useful for comparing models in binary classification tasks, especially when class distributions are balanced. However, it may not be as informative as the Precision-Recall curve in highly imbalanced datasets.

As a graph a y=x line would represent random classifier while y=1 line would be ideal classifier. [This](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) article has excellent explanation for ROC and PR curves. I encourage you to read it. 

##### PR Curve

A **Precision-Recall (PR) curve** is a graphical representation that illustrates the trade-off between precision and recall for different threshold values in a binary classification task. 

- **Precision** (y-axis) is the ratio of true positive predictions to the total number of positive predictions (true positives + false positives).
- **Recall** (x-axis), also known as sensitivity, is the ratio of true positive predictions to the total number of actual positive instances (true positives + false negatives).

The PR curve is particularly useful for evaluating the performance of models on imbalanced datasets, where the positive class is of greater interest. It helps in understanding how well a model can identify positive instances while minimizing false positives.

### Experiment Setup
The experiment involves evaluating the performance of binary classifiers under varying conditions of class imbalance and prediction probabilities. A synthetic dataset is generated with 10,000 samples, where the true class labels (`y_true`) are created based on specified imbalance ratios (0.5, 0.1, 0.9), representing balanced, minority, and majority class scenarios, respectively. Prediction scores (`y_scores`) are generated with probabilities of predicting class 1 set to 0 (biased), 0.5 (random), and 1 (biased).

For each combination of probability and imbalance, key performance metrics are computed, including accuracy, precision, recall, F1-score, and AUROC. Confusion matrices are constructed to visualize the distribution of true positives, false positives, true negatives, and false negatives. Precision-Recall (PR) and ROC curves are plotted to assess the trade-offs between precision and recall, and the ability to differentiate between classes across thresholds.

The results are visualized for confusion matrices, ROC/PR curves and classification reports, providing a comprehensive view of classifier performance under different scenarios. The aim is to understand how class imbalance and prediction biases affect various evaluation metrics, offering insights into model robustness and reliability.


### PR Curve Analysis for Different Classifier Types and Class Imbalance
The Precision-Recall (PR) curve provides insights into the performance of classifiers, especially in the context of imbalanced datasets. Here's an explanation of the PR curve behavior for the three classifier types—random, all 1, and all 0—and the impact of class imbalance:

### Classifier Types

1. **Random Classifier**:
   - **Behavior**: The PR curve for a random classifier typically hovers around the baseline precision, which is the ratio of positive instances in the dataset. This means that precision is equivalent to the class imbalance ratio.
   - **Impact of Imbalance**: As class imbalance increases, the baseline precision decreases, making the PR curve appear lower. The random classifier doesn't perform better than chance, so its curve is flat and close to the baseline.

2. **All 1 Classifier**:
   - **Behavior**: This classifier predicts all instances as positive. The recall is always 1, but precision varies based on the number of true positives versus false positives.
   - **Impact of Imbalance**: With severe class imbalance (more negatives), precision drops significantly because most predictions are false positives. The PR curve starts at the point (1, precision) and is a horizontal line.

3. **All 0 Classifier**:
   - **Behavior**: This classifier predicts all instances as negative. It doesn't contribute to the PR curve since recall is 0 (no true positives).
   - **Impact of Imbalance**: The PR curve is not meaningful for this classifier as it does not predict any positive class.

### Impact of Class Imbalance

- **Sensitivity to Imbalance**: The PR curve is highly sensitive to class imbalance. Unlike ROC curves, which remain unchanged, PR curves shift significantly with changes in class distribution[2][5].
- **Interpretation**: In imbalanced datasets, the PR curve is more informative than the ROC curve because it focuses on the minority class, providing a clearer picture of a classifier's ability to predict positive instances[4][5].

### Conclusion

The PR curve is a valuable tool for evaluating classifiers on imbalanced datasets. It highlights the trade-off between precision and recall, making it easier to understand the classifier's performance in identifying the minority class. The behavior of the PR curve for random, all 1, and all 0 classifiers illustrates how class imbalance affects precision and recall, emphasizing the need for careful interpretation in such scenarios.

### AUROC Analysis for Different Classifier Types and Class Imbalance

The Area Under the Receiver Operating Characteristic (AUROC) curve is a common metric for evaluating the performance of classifiers. It measures the ability of a model to distinguish between classes. Here's how AUROC behaves for different classifier types and the impact of class imbalance:

#### Classifier Types

1. **Random Classifier**:
   - **Behavior**: The AUROC for a random classifier is typically around 0.5, indicating no discriminative power. This means the classifier is guessing randomly, similar to flipping a coin.
   - **Impact of Imbalance**: Class imbalance does not affect the AUROC of a random classifier, as it remains at 0.5 regardless of the class distribution.

2. **All 1 Classifier**:
   - **Behavior**: This classifier predicts all instances as positive. The AUROC is not meaningful here because the model does not provide any true negatives, making it impossible to compute a meaningful false positive rate.
   - **Impact of Imbalance**: The AUROC might be misleadingly high if the positive class is the minority, as the classifier captures all positives but also misclassifies all negatives.

3. **All 0 Classifier**:
   - **Behavior**: This classifier predicts all instances as negative. Similar to the all 1 classifier, the AUROC is not meaningful because the model does not provide any true positives.
   - **Impact of Imbalance**: If the negative class is the majority, the AUROC might appear deceptively high due to the abundance of true negatives.

#### Impact of Class Imbalance

- **Sensitivity to Imbalance**: Unlike the PR curve, the AUROC is less sensitive to class imbalance. It evaluates the model's ability to rank positive instances higher than negative ones, regardless of their proportions in the dataset.
- **Interpretation**: In highly imbalanced datasets, AUROC can be overly optimistic. This is because the false positive rate (FPR) is influenced more by the large number of true negatives, making it easier to achieve a low FPR even if the model is not effectively identifying the minority class.

### Conclusion

AUROC is a robust metric for balanced datasets but can be misleading in imbalanced scenarios. It provides a general sense of a model's ranking ability but may not reflect the actual performance in identifying minority classes. In such cases, the Precision-Recall curve might offer more relevant insights.

### Accuracy Analysis for Different Classifier Types and Class Imbalance

Accuracy is a straightforward metric that measures the proportion of correct predictions. However, its usefulness can vary significantly depending on the classifier type and class imbalance.

#### Classifier Types

1. **Random Classifier**:
   - **Behavior**: The accuracy of a random classifier depends on the class distribution. It will be close to the proportion of the majority class, as it randomly guesses the class labels.
   - **Impact of Imbalance**: With increased class imbalance, accuracy may appear deceptively high because the classifier often guesses the majority class correctly.

2. **All 1 Classifier**:
   - **Behavior**: This classifier predicts all instances as positive. Its accuracy is equal to the proportion of the positive class in the dataset.
   - **Impact of Imbalance**: If the positive class is the minority, accuracy will be low. Conversely, if the positive class is the majority, accuracy will be high, but this doesn't reflect true performance.

3. **All 0 Classifier**:
   - **Behavior**: This classifier predicts all instances as negative. Its accuracy is equal to the proportion of the negative class in the dataset.
   - **Impact of Imbalance**: If the negative class is the majority, accuracy will be high, but this is misleading as the classifier fails to identify any positive instances.

#### Impact of Class Imbalance

- **Sensitivity to Imbalance**: Accuracy is highly sensitive to class imbalance. It can be misleading in imbalanced datasets because it doesn't account for the distribution of classes. A high accuracy might simply reflect the majority class's prevalence rather than the classifier's ability to correctly identify both classes.
- **Interpretation**: In imbalanced scenarios, accuracy often overestimates the performance of classifiers that predict the majority class well but fail on the minority class.

### Conclusion

While accuracy is a useful metric in balanced datasets, it can be misleading in imbalanced situations. It does not provide insight into the classifier's ability to correctly identify minority class instances, making it less informative than metrics like precision, recall, or the PR curve in such contexts.

### Precision Analysis for Different Classifier Types and Class Imbalance

Precision is a metric that measures the proportion of true positive predictions among all positive predictions. Here's how precision behaves for different classifier types and the impact of class imbalance:

#### Classifier Types

1. **Random Classifier**:
   - **Behavior**: Precision for a random classifier is generally low and hovers around the baseline, which is the proportion of positive instances in the dataset.
   - **Impact of Imbalance**: As class imbalance increases (more negatives), precision decreases because the classifier makes more false positive predictions relative to true positives.

2. **All 1 Classifier**:
   - **Behavior**: This classifier predicts all instances as positive. Precision depends on the ratio of true positives to total positive predictions (which are all predictions in this case).
   - **Impact of Imbalance**: Precision is low if the positive class is the minority because there are many false positives. If the positive class is the majority, precision improves but still doesn't reflect the classifier's ability to distinguish classes.

3. **All 0 Classifier**:
   - **Behavior**: This classifier predicts all instances as negative, so precision is undefined as there are no positive predictions.
   - **Impact of Imbalance**: Precision is not applicable here since the classifier doesn't predict any positives.

#### Impact of Class Imbalance

- **Sensitivity to Imbalance**: Precision is highly sensitive to class imbalance. In datasets with a large negative class, precision can be misleadingly low if the classifier predicts many false positives.
- **Interpretation**: High precision indicates a low false positive rate, which is crucial in scenarios where false positives are costly. However, it must be considered alongside recall to get a full picture of performance, especially in imbalanced datasets.

### Conclusion

Precision is a valuable metric for understanding a classifier's performance in terms of false positives, but it must be interpreted with caution in imbalanced datasets. It provides insight into the accuracy of positive predictions but should be used in conjunction with recall to assess overall effectiveness.

### Recall Analysis for Different Classifier Types and Class Imbalance

Recall, also known as sensitivity or true positive rate, measures the proportion of actual positives correctly identified. Here's how recall behaves for different classifier types and the impact of class imbalance:

#### Classifier Types

1. **Random Classifier**:
   - **Behavior**: Recall for a random classifier is generally low, as it randomly predicts class labels without a specific focus on capturing positives.
   - **Impact of Imbalance**: Class imbalance does not directly affect recall, but the random nature means it often misses many positives, leading to low recall.

2. **All 1 Classifier**:
   - **Behavior**: This classifier predicts all instances as positive, achieving a recall of 1.0 because it captures all true positives.
   - **Impact of Imbalance**: Recall remains perfect regardless of class imbalance, but this doesn't reflect true performance since it also includes many false positives.

3. **All 0 Classifier**:
   - **Behavior**: This classifier predicts all instances as negative, resulting in a recall of 0.0 because it fails to identify any positives.
   - **Impact of Imbalance**: Recall is always zero, regardless of the class distribution, as no positive predictions are made.

#### Impact of Class Imbalance

- **Sensitivity to Imbalance**: Recall is less sensitive to class imbalance compared to precision. It focuses solely on the ability to identify positives, regardless of the number of negatives.
- **Interpretation**: High recall indicates the model's effectiveness in capturing positives, which is crucial in scenarios where missing positive instances is costly. However, it should be balanced with precision to avoid a high false positive rate.

### Conclusion

Recall is a vital metric for assessing a classifier's ability to identify positive instances, especially in imbalanced datasets. However, it should be considered alongside precision to provide a complete picture of the model's performance. The PR curve in the image illustrates these trade-offs, showing how different scenarios affect precision and recall.

### F1 Score Analysis for Different Classifier Types and Class Imbalance

The F1 score is the harmonic mean of precision and recall, providing a balance between the two. Here's how the F1 score behaves for different classifier types and the impact of class imbalance:

#### Classifier Types

1. **Random Classifier**:
   - **Behavior**: The F1 score for a random classifier is generally low, reflecting its inability to consistently identify true positives or negatives.
   - **Impact of Imbalance**: Class imbalance exacerbates the low F1 score, as random predictions lead to low precision and recall.

2. **All 1 Classifier**:
   - **Behavior**: This classifier predicts all instances as positive. It achieves a high recall but low precision, resulting in a moderate F1 score.
   - **Impact of Imbalance**: If the positive class is the minority, the F1 score is low due to poor precision. If the positive class is the majority, the F1 score improves but still doesn't reflect true discrimination ability.

3. **All 0 Classifier**:
   - **Behavior**: This classifier predicts all instances as negative, resulting in an F1 score of zero because recall is zero.
   - **Impact of Imbalance**: The F1 score remains zero regardless of class distribution, as no positives are predicted.

#### Impact of Class Imbalance

- **Sensitivity to Imbalance**: The F1 score is sensitive to class imbalance, as it depends on both precision and recall. In imbalanced datasets, a high F1 score is more challenging to achieve because of the difficulty in maintaining both high precision and recall.
- **Interpretation**: The F1 score provides a single metric that balances the trade-off between precision and recall, making it useful in scenarios where both false positives and false negatives are important.

### Conclusion

The F1 score is a valuable metric for evaluating classifiers, especially in imbalanced datasets. It considers both precision and recall, providing a more comprehensive view of performance. However, it should be interpreted in the context of specific application needs, as it may not fully capture the nuances of class distribution.

### Confusion Matrix Analysis for Different Classifier Types and Class Imbalance

The confusion matrix is a tool that summarizes the performance of a classification algorithm by showing the counts of true positives, false positives, true negatives, and false negatives. Here's how it behaves for different classifier types and the impact of class imbalance:

#### Classifier Types

1. **Random Classifier**:
   - **Behavior**: The confusion matrix for a random classifier shows a mix of true and false positives and negatives. The distribution depends on class proportions.
   - **Impact of Imbalance**: With increased imbalance, the matrix will show more false negatives or false positives, depending on the majority class, as random guessing aligns more with the prevalent class.

2. **All 1 Classifier**:
   - **Behavior**: This classifier predicts all instances as positive, resulting in zero true negatives and many false positives.
   - **Impact of Imbalance**: If the positive class is the minority, the confusion matrix will show a high number of false positives. If the positive class is the majority, it will show more true positives but still lacks true negatives.

3. **All 0 Classifier**:
   - **Behavior**: This classifier predicts all instances as negative, resulting in zero true positives and many false negatives.
   - **Impact of Imbalance**: If the negative class is the majority, the confusion matrix will show a high number of true negatives. If the negative class is the minority, it will show more false negatives.

#### Impact of Class Imbalance

- **Sensitivity to Imbalance**: The confusion matrix is highly sensitive to class imbalance. It clearly shows the skew in predictions, highlighting whether the classifier is biased towards the majority class.
- **Interpretation**: The matrix provides a detailed view of how predictions are distributed across classes, making it easier to identify where a classifier may be failing, especially in imbalanced datasets.

### Conclusion

The confusion matrix is a valuable diagnostic tool for understanding classifier performance, particularly in imbalanced datasets. It provides a clear picture of where errors are occurring, allowing for targeted improvements in model performance.

### Confusion Matrices Analysis

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

Here's a summary table for the metrics—precision, recall, F1-score, AUROC, AUC PR curve, and accuracy—incorporating class imbalance and the three types of classifiers: random, all 1, and all 0.

| **Metric**   | **Pros**                                                                 | **Cons**                                                               | **Class Imbalance Impact**                                           | **Random Classifier Value** | **Recommendation**                                               |
|--------------|--------------------------------------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------|-----------------------------|------------------------------------------------------------------|
| **Precision**| High precision reduces false positives                                   | Can be low if false positives are high                                 | Decreases with more false positives in imbalanced datasets           | Low                         | Use when false positives are costly                              |
| **Recall**   | High recall captures more true positives                                 | Can be high with many false positives                                  | Insensitive to imbalance, but may miss positives if not balanced     | Low                         | Use when false negatives are costly                              |
| **F1-score** | Balances precision and recall                                            | May not reflect true performance if one metric is very low             | Sensitive to imbalance, hard to achieve high values                  | Low                         | Use when both false positives and negatives matter               |
| **AUROC**    | Measures ranking ability across thresholds                               | Can be misleadingly high in imbalanced datasets                        | Less sensitive to imbalance, but may not reflect minority class      | 0.5                         | Use for balanced datasets                                        |
| **AUC PR**   | Focuses on positive class performance                                    | More sensitive to class imbalance than AUROC                           | Provides clearer picture in imbalanced datasets                      | Low                         | Prefer over AUROC in imbalanced datasets                         |
| **Accuracy** | Simple and intuitive                                                     | Can be misleading in imbalanced datasets                               | Overestimates performance if majority class is predicted correctly   | Depends on class distribution | Use cautiously, not recommended for imbalanced datasets          |

### Key Points

- **Precision** is useful when false positives are costly, but it can be misleading in imbalanced datasets.
- **Recall** is crucial when missing true positives is costly, but it should be balanced with precision.
- **F1-score** provides a balance between precision and recall, useful when both are important.
- **AUROC** is less sensitive to imbalance but may not reflect true performance for minority classes.
- **AUC PR** is more informative in imbalanced scenarios, focusing on the positive class.
- **Accuracy** can be deceptive in imbalanced datasets, as it may reflect majority class prevalence rather than true performance.

This table helps in choosing the right metric based on the specific needs and characteristics of the dataset and the classifier.

Citations:
[1] https://pplx-res.cloudinary.com/image/upload/v1724448133/user_uploads/utcigjrpn/pr_curves.jpg

### References
[1] https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
[2] https://en.wikipedia.org/wiki/Precision_and_recall
[3] https://developers.google.com/machine-learning/testing-debugging/metrics/metrics
[4] https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
[5] https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/
[6] https://en.wikipedia.org/wiki/Receiver_operating_characteristic
[7] https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5?gi=6493ad0a1a35
[8] https://link.springer.com/referenceworkentry/10.1007/978-1-4419-9863-7_209
[9] https://h2o.ai/wiki/auc-roc/
[10] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3755824/