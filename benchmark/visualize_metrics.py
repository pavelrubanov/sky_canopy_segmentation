# -----------------------------------------------------------------------------
# Визуализация метрик
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from PIL.ImagePath import Path


def visualize_metrics(metrics, figsize=(20, 5),
                      save_path: str | None = './metrics'):
    """
    Рисует четыре сюжета:
      1) столбец со смещением (bias);
      2) столбец с MAE;
      3) диаграмма рассеяния «предсказано vs. истина»;
      4) график Бланда–Альтмана.

    Parameters
    ----------
    metrics      : словарь, который возвращает функция evaluate(...)
    figsize      : размер полотна фигуры в дюймах
    save_path    : если указан — PNG будет сохранён по этому пути
    """
    per_img = metrics["per_image"]

    # --- данные --------------------------------------------------------------
    gt_vals = np.array([d["gt_percent"] for d in per_img], dtype=float)
    pr_vals = np.array([d["pred_percent"] for d in per_img], dtype=float)
    diff_vals = pr_vals - gt_vals
    mean_pair = (gt_vals + pr_vals) / 2.0

    bias_val = metrics["bias_percent"]
    mae_val = metrics["mae_percent"]

    fig, axes = plt.subplots(1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 3, 3]}
)

    # --- 1. Бар: смещение ----------------------------------------------------
    axes[0].bar([0], [bias_val], width=0.1, color="steelblue")
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["Смещение, %"])
    axes[0].set_ylabel("Значение (процентные пункты)")
    axes[0].set_ylim(-1, 1)
    axes[0].set_title("Смещение (bias)")

    # --- 2. Бар: MAE ---------------------------------------------------------
    axes[1].bar([0], [mae_val], width=0.1, color="darkorange")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["MAE, п.п."])
    axes[1].set_ylim(0, 3)
    axes[1].set_title("Средняя абсолютная\nошибка (MAE)")

    # --- 3. Предсказано vs Истина -------------------------------------------
    axes[2].scatter(gt_vals, pr_vals, alpha=0.6)
    axes[2].plot([0, 25], [0, 25], ls="--")      # линия y = x
    axes[2].set_xlabel("Истинная доля просвета, %")
    axes[2].set_ylabel("Предсказанная доля просвета, %")
    axes[2].set_title(f"Предсказание vs истина\nMAE = {mae_val:.2f} п.п.")

    # --- 4. График Бланда–Альтмана ------------------------------------------
    ba = metrics["bland_altman"]
    axes[3].scatter(mean_pair, diff_vals, alpha=0.6)
    axes[3].axhline(ba["mean_diff"], ls="--", label="смещение")
    axes[3].axhline(ba["loa_lower"], ls="--", color="red", label="границы")
    axes[3].axhline(ba["loa_upper"], ls="--", color="red")
    axes[3].set_xlabel("Среднее GT и предсказания, %")
    axes[3].set_ylabel("Предсказание − GT, п.п.")
    axes[3].set_title("График Бланда–Альтмана")
    axes[3].legend(frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()