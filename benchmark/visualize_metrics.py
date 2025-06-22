# -----------------------------------------------------------------------------
# Visualisation helper
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from PIL.ImagePath import Path


def visualize_metrics(metrics, figsize = (15, 5),
                      save_path = './metrics'):
    """
    Рисует три классических сюжета:
      1) распределение IoU (box-plot);
      2) scatter «предсказано vs. истина» с диагональю и подписями MAE;
      3) график Bland–Altman c средним смещением и пределами согласия ±1.96 SD.

    Parameters
    ----------
    metrics      : словарь, который возвращает функция evaluate(...)
    figsize      : размер полотна фигуры в дюймах
    save_path    : если указан — PNG будет сохранён по этому пути
    """
    per_img = metrics["per_image"]
    iou_vals = np.array([d["iou"] for d in per_img], dtype=float)
    gt_vals  = np.array([d["gt_percent"]  for d in per_img], dtype=float)
    pr_vals  = np.array([d["pred_percent"] for d in per_img], dtype=float)
    diff_vals = pr_vals - gt_vals
    mean_pair = (gt_vals + pr_vals) / 2.0

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- 1. IoU distribution -------------------------------------------------
    axes[0].boxplot(iou_vals, vert=True, showfliers=False)
    axes[0].set_ylabel("IoU")
    axes[0].set_title(f"IoU distribution\nmean = {metrics['mean_iou']:.3f}")

    # --- 2. Predicted vs Ground Truth ---------------------------------------
    axes[1].scatter(gt_vals, pr_vals, alpha=0.6)
    axes[1].plot([0, 25], [0, 25], ls="--")      # y = x
    axes[1].set_xlabel("Ground truth gap fraction, %")
    axes[1].set_ylabel("Predicted gap fraction, %")
    axes[1].set_title(f"Prediction vs GT\nMAE = {metrics['mae_percent']:.2f} pp")

    # --- 3. Bland–Altman plot ------------------------------------------------
    ba = metrics["bland_altman"]
    axes[2].scatter(mean_pair, diff_vals, alpha=0.6)
    axes[2].axhline(ba["mean_diff"],   ls="--", label="bias")
    axes[2].axhline(ba["loa_lower"],   ls="--", color="red",  label="limits")
    axes[2].axhline(ba["loa_upper"],   ls="--", color="red")
    axes[2].set_xlabel("Mean of GT and prediction, %")
    axes[2].set_ylabel("Prediction − GT, % points")
    axes[2].set_title("Bland–Altman plot")
    axes[2].legend(frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()
