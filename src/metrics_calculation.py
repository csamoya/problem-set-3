'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for genre in genre_list:
        tp = genre_tp_counts.get(genre, 0)
        fp = genre_fp_counts.get(genre, 0)
        fn = genre_true_counts.get(genre, 0) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0  

    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list

    # Your code here

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    pre_rows = []
    true_rows = []

    for _, row in model_pred_df.iterrows():
        true_genres = [g.strip().strip("[]'\"") for g in row['true_genres'].split(',')]
        pred_genres = [g.strip().strip("[]'\"") for g in row['predicted'].split(',')]

        true_row = [1 if genre in true_genres else 0 for genre in genre_list]
        pred_row = [1 if genre in pred_genres else 0 for genre in genre_list]

        true_rows.append(true_row)
        pre_rows.append(pred_row)   

    true_matrix = pd.DataFrame(true_rows, columns=genre_list)
    pred_matrix = pd.DataFrame(pre_rows, columns=genre_list)

    precision, recall, f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='macro', zero_division=0)
    macro_prec = precision
    macro_rec = recall
    macro_f1 = f1

    precision, recall, f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='micro', zero_division=0)
    micro_prec = precision
    micro_rec = recall
    micro_f1 = f1

    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1

    # Your code here
