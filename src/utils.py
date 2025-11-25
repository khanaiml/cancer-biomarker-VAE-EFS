import os
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def calculate_metrics(conf_matrix, y_true, y_pred):
    TP = np.diag(conf_matrix).astype(float)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP
    TN = conf_matrix.sum() - (TP + FP + FN)
    eps = 1e-10
    
    TPR = TP / (TP + FN + eps)
    TNR = TN / (TN + FP + eps)
    PPV = TP / (TP + FP + eps)
    NPV = TN / (TN + FN + eps)
    F1S = 2 * (PPV * TPR) / (PPV + TPR + eps)
    FPR = FP / (FP + TN + eps)
    FNR = FN / (TP + FN + eps)
    FDR = FP / (TP + FP + eps)
    ACC = (TP + TN) / (TP + TN + FP + FN + eps)
    OA = round(accuracy_score(y_true, y_pred), 4)
    
    return {
        'TP': TP.tolist(),
        'TN': TN.tolist(),
        'FP': FP.tolist(),
        'FN': FN.tolist(),
        'Confusion Matrix': conf_matrix,
        'Sensitivity (TPR)': [round(x, 4) for x in TPR],
        'Specificity (TNR)': [round(x, 4) for x in TNR],
        'Precision (PPV)': [round(x, 4) for x in PPV],
        'NPV': [round(x, 4) for x in NPV],
        'F1 Score': [round(x, 4) for x in F1S],
        'FPR': [round(x, 4) for x in FPR],
        'FNR': [round(x, 4) for x in FNR],
        'FDR': [round(x, 4) for x in FDR],
        'Classwise Accuracy': [round(x, 4) for x in ACC],
        'Overall Accuracy': OA
    }

def display_metrics(metrics, class_names):
    print("\n" + "="*80)
    print("   " + "\t".join(class_names))
    print("   " + "-" * (8 * len(class_names)))
    
    for key, value in metrics.items():
        if key == 'Confusion Matrix':
            continue
        elif key == 'Overall Accuracy':
            print(f"\n{key}: {value:.4f}")
        else:
            formatted = "\t".join([f"{x:.4f}" if isinstance(x, float) else str(x) for x in value])
            print(f"{formatted}   :: {key}")
    print("="*80 + "\n")