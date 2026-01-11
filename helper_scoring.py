from sklearn.metrics import accuracy_score, recall_score, precision_score, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def get_scores(y_test, y_pred, cut=0.5, rounding=4):

    y_predicted_classes = np.where(y_pred >= cut, 1, 0)

    # Average Precision Score
    avg_prec_score = round(average_precision_score(y_test, y_pred), rounding)

    # ROC
    roc_score = round(roc_auc_score(y_test, y_pred), rounding)

    # Recall Score
    rec_score = round(recall_score(y_test, y_predicted_classes), rounding)

    # precision Score
    prec_score = round(precision_score(y_test, y_predicted_classes), rounding)

    # confusion matrix
    cm = confusion_matrix(y_test, y_predicted_classes)
    
    # False Positives share
    fp_share = round(cm[0][1] / cm.sum(), rounding)

    # accury
    acc = round(accuracy_score(y_test, y_predicted_classes), rounding)

    return {"accuracy":acc, "avg_prec_score":avg_prec_score, "roc_score":roc_score, "rec_score":rec_score, "prec_score":prec_score, "fp_share":fp_share, "cm":cm}

def plot_pr_curve(y_true, y_pred):
    pr = precision_recall_curve(y_true, y_pred)
    plt.plot(pr[1], pr[0], label="precision-recall curve")
    plt.axhline(y_true.mean(), label="no-skill line")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
