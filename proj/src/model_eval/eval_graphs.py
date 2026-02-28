import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd


def plot_model_diagnostics(y_true, y_prob, stats_per_thresh, model_name, out_dir):
    save_path = os.path.join(out_dir, f"{model_name}_diagnostics.png")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Model Diagnostics: {model_name}", fontsize=16)

    # --- 1. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange',
                 lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([-0.01, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc="lower right")

    # --- 2. Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, color='green',
                 lw=2, label=f'PR-AUC = {pr_auc:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")

    # --- 3. Metrics vs. Threshold (F2) ---
    thresholds = [s['threshold'] for s in stats_per_thresh]
    precs = [s['precision'] for s in stats_per_thresh]
    recs = [s['recall'] for s in stats_per_thresh]
    f2s = [s['f2'] for s in stats_per_thresh]

    axes[2].plot(thresholds, precs, label='Precision',
                 linestyle=':', alpha=0.6, color='gray')
    axes[2].plot(thresholds, recs, label='Recall',
                 linestyle='--', alpha=0.6, color='blue')
    axes[2].plot(thresholds, f2s, label='F2 Score',
                 linewidth=2.5, color='black')
    max_f2_idx = np.argmax(f2s)
    best_thresh = thresholds[max_f2_idx]
    best_score = f2s[max_f2_idx]

    axes[2].axvline(best_thresh, color='red', linestyle='-', alpha=0.5,
                    label=f'Opt Thresh (F2): {best_thresh:.2f}')

    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('Score')
    axes[2].set_title(f'Metrics vs. Threshold (Max F2: {best_score:.3f})')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(all_models_data, summary_df, out_dir):
    sns.set_style("whitegrid")

    # --- 1. Combined ROC & PR Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = sns.color_palette("husl", len(all_models_data))

    for (model_name, data), color in zip(all_models_data.items(), colors):
        y_true = data['y_true']
        y_prob = data['y_prob']

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2, alpha=0.8,
                     label=f'{model_name} (AUC={roc_auc:.3f})')

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, color=color, lw=2, alpha=0.8,
                     label=f'{model_name} (PR={pr_auc:.3f})')

    # Formatting ROC
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Combined ROC Curves')
    axes[0].legend(loc="lower right", fontsize='small')

    # Formatting PR
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Combined Precision-Recall Curves')
    axes[1].legend(loc="lower left", fontsize='small')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_curves.png"), dpi=300)
    plt.close()

    # --- 2. Metric Bar Chart Comparison ---
    comparison_data = []

    for _, row in summary_df.iterrows():
        model = row['model']
        comparison_data.append(
            {'Model': model, 'Metric': 'Max F2', 'Value': row['f2']})
        comparison_data.append(
            {'Model': model, 'Metric': 'PR-AUC', 'Value': row['pr_auc']})
        if 'recall_fpr_1' in row:
            comparison_data.append(
                {'Model': model, 'Metric': 'Recall @ 1% FPR', 'Value': row['recall_fpr_1']})

    comp_df = pd.DataFrame(comparison_data)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=comp_df, x='Model', y='Value',
                hue='Metric', palette='viridis')
    plt.title('Key Metrics Comparison (F2 Optimized)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_metrics_bar.png"), dpi=300)
    plt.close()
