import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import os
import polars as pl
import altair as alt

# Create 'charts' directory if it doesn't exist
if not os.path.exists("charts"):
    os.makedirs("charts")


def generate_data(n_samples, imbalance, p):
    if imbalance == 0.5:
        y_true = np.random.choice([0, 1], size=n_samples)
    else:
        n_ones = int(n_samples * imbalance)
        n_zeros = n_samples - n_ones
        y_true = np.concatenate([np.ones(n_ones), np.zeros(n_zeros)])
        np.random.shuffle(y_true)
    y_scores = np.random.choice([0, 1], size=n_samples, p=[1 - p, p])
    return y_true, y_scores


def calculate_metrics(y_true, y_scores):
    accuracy = accuracy_score(y_true, y_scores)
    precision = precision_score(y_true, y_scores, zero_division=0)
    recall = recall_score(y_true, y_scores, zero_division=0)
    f1 = f1_score(y_true, y_scores, zero_division=0)
    auroc = roc_auc_score(y_true, y_scores)
    return accuracy, precision, recall, f1, auroc


n_samples = 10000
ps = [0, 0.5, 1]
imbalances = [0.5, 0.1, 0.9]
results = []
report_data = []
roc_curves = []
pr_curves = []
confusion_matrices = []

for p in ps:
    for imbalance in imbalances:
        y_true, y_scores = generate_data(n_samples, imbalance, p)
        metrics = calculate_metrics(y_true, y_scores)
        accuracy = metrics[0]
        key = f"probability={p}, imbalance={imbalance}"
        results.append((key, p, imbalance, *metrics))

        # Confusion Matrix with numbers
        cm = confusion_matrix(y_true, y_scores)
        tn, fp, fn, tp = cm.ravel()
        confusion_matrices.append(
            {
                "imbalance": imbalance,
                "probability_class_1": p,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )
        # Classification Report
        report = classification_report(
            y_true, y_scores, zero_division=0, output_dict=True
        )
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                try:
                    label_name = f"Class {int(float(label))}"
                except ValueError:
                    label_name = label.replace(" avg", " average")
                report_data.append(
                    {
                        "probability_of_class_1": p,
                        "imbalance": imbalance,
                        "type": label_name,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1-score": metrics["f1-score"],
                        "accuracy": accuracy,
                        "support": metrics["support"],
                    }
                )

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_curves.append((fpr, tpr, key))

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_curves.append((precision, recall, key))

# Plotting the confusion matrices
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

for i, p in enumerate(ps):
    for j, imbalance in enumerate(imbalances):
        ax = axes[i, j]
        cm_data = [
            x
            for x in confusion_matrices
            if x["probability_class_1"] == p and x["imbalance"] == imbalance
        ][0]
        tn, fp, fn, tp = cm_data["TN"], cm_data["FP"], cm_data["FN"], cm_data["TP"]
        cm = np.array([[tn, fp], [fn, tp]])

        cax = ax.matshow(cm, cmap="Blues")
        ax.set_title(f"p={p}, imbalance={imbalance}", pad=20)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for (k, l), val in np.ndenumerate(cm):
            ax.text(
                l,
                k,
                f"{val}",
                ha="center",
                va="center",
                color="red",
                fontsize="xx-large",
            )

# Adjust layout manually
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.savefig("charts/confusion_matrices.png")
plt.close()

# Overlay ROC Curves
plt.figure(figsize=(8, 6))
for fpr, tpr, label in roc_curves:
    plt.plot(fpr, tpr, label=label)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Different Scenarios")
plt.legend(loc="lower right", fontsize="small")
plt.savefig("charts/roc_curves.png")
plt.close()

# Overlay Precision-Recall Curves
plt.figure(figsize=(8, 6))
for precision, recall, label in pr_curves:
    plt.plot(recall, precision, label=label)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for Different Scenarios")
plt.legend(loc="lower right", fontsize="small")
plt.savefig("charts/pr_curves.png")
plt.close()

# Create a joint table for classification reports
joint_report_df = pl.DataFrame(report_data)

# Convert Polars DataFrame to Pandas for Altair compatibility
df = joint_report_df

for metric in "precision recall f1-score accuracy".split():
    # Create an Altair chart with both probability_of_class_1 and imbalance
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="type:N",
            y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 1])),
            color="type:N",
        )
        .facet(column="probability_of_class_1:O", row="imbalance:O")
    )
    chart.save(f"charts/{metric}_chart.png")
