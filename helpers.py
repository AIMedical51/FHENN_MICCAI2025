import torch
import scipy.ndimage as nd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import numpy as np
def compute_ece(labels, preds, probs, n_bins=15):

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    confidences = np.max(probs, axis=1)  
    
   
    accuracies = (preds == labels).astype(float)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= lower) & (confidences < upper)
        
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)

    return ece
def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    device = y.device
    labels = labels.to(device)

    return y[labels]


def rotate_img(x, deg):
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()


def compute_metrics_per_class(y_true, y_pred, num_classes):
    """
    Compute Accuracy for each class and overall Accuracy.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tp = np.diag(cm)  # True Positives for each class
    fn = cm.sum(axis=1) - tp  # False Negatives for each class
    class_acc = tp / (tp + fn + 1e-8)  # Per-class accuracy
    overall_acc = tp.sum() / cm.sum()  # Overall accuracy
    return class_acc, overall_acc
def compute_metrics(y_true, y_pred, num_classes):
    """
    Compute Accuracy, Sensitivity, Specificity, and F1-score.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tp = np.diag(cm)  # True Positives for each class
    fp = cm.sum(axis=0) - tp  # False Positives for each class
    fn = cm.sum(axis=1) - tp  # False Negatives for each class
    tn = cm.sum() - (fp + fn + tp)  # True Negatives for each class
    
    # Sensitivity and Specificity per class
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    # Weighted average metrics
    accuracy = tp.sum() / cm.sum()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return accuracy, sensitivity.mean(), specificity.mean(), f1
