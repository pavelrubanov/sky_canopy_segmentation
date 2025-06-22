from typing import Callable, Iterable, List, Tuple, Dict, Any
import numpy as np
from pathlib import Path
from utils import calculate_real_percentage
from PIL import Image
from pprint import pprint
from src.predict import process, resize_large_image
from visualize_metrics import visualize_metrics

# -----------------------------------------------------------------------------
# Pixel-level helpers
# -----------------------------------------------------------------------------

def compute_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    # Edge case: both masks have no positive pixels → perfect IoU = 1
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Dice coefficient (F1-score for sets).

    Dice = 2⋅|A ∩ B| / (|A| + |B|)

    Edge case: если обе маски пустые, Dice = 1.
    """
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    size_sum = gt.sum() + pred.sum()

    if size_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / size_sum

# -----------------------------------------------------------------------------
# Dataset-level evaluation
# -----------------------------------------------------------------------------

def evaluate(
    process,
    dataset
) -> Dict[str, Any]:
    """
    metrics : dict
        Keys:
            • mean_iou           — average IoU over all images
            • mean_dice          — average Dice over all images
            • mae_percent        — mean absolute error of percent (|Δ|)
            • bias_percent       — mean signed error of percent (Δ)
            • bland_altman       — dict with mean_diff, loa_lower, loa_upper
            • per_image          — list of per-image results (index, iou, dice…)
    """
    ious: List[float] = []
    dices: List[float] = []
    abs_errs: List[float] = []
    signed_errs: List[float] = []
    details: List[Dict[str, Any]] = []

    for img, gt_mask, gt_percent in dataset:
        print(f"Processing image {img.stem}...")
        pred_percent, pred_mask = process(img)

        # --- pixel-level metrics ---
        iou = compute_iou(gt_mask, pred_mask)
        dice = compute_dice(gt_mask, pred_mask)
        ious.append(iou)
        dices.append(dice)

        # --- gap-fraction errors (in % points) ---
        diff = float(pred_percent) - float(gt_percent)
        abs_errs.append(abs(diff))
        signed_errs.append(diff)

        details.append({
            "iou": iou,
            "dice": dice,
            "gt_percent": float(gt_percent),
            "pred_percent": float(pred_percent),
            "error": diff,
        })

    ious_np = np.array(ious, dtype=float)
    dices_np = np.array(dices, dtype=float)
    abs_np = np.array(abs_errs, dtype=float)
    diff_np = np.array(signed_errs, dtype=float)

    mean_iou: float = float(ious_np.mean())
    mean_dice: float = float(dices_np.mean())
    mae: float = float(abs_np.mean())
    bias: float = float(diff_np.mean())

    # Bland–Altman limits of agreement (95 %)
    std_diff: float = float(diff_np.std(ddof=1))  # sample std-dev (n-1)
    loa_lower: float = bias - 1.96 * std_diff
    loa_upper: float = bias + 1.96 * std_diff

    return {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "mae_percent": mae,
        "bias_percent": bias,
        "bland_altman": {
            "mean_diff": bias,
            "loa_lower": loa_lower,
            "loa_upper": loa_upper,
        },
        "per_image": details,
    }

def local_process(img):
    return process(img, tile_size=540, model_path='../src/imageseg_canopy_model.hdf5', save=False)

# -----------------------------------------------------------------------------
# Simple CLI demo (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = Path('data')  # Папка с изображениями в формате jpg
    masks_dir = Path('masks')  # Папка с масками в формате png
    image_paths = sorted(data_dir.glob('*.jpg'))
    data = []
    for img in image_paths:
        mask_path = masks_dir / f'{img.stem}.png'
        if not mask_path.exists():
            print(f"Пропуск image_path: маска {mask_path} не найдена")
            continue

        gt_mask_img = resize_large_image(Image.open(mask_path))
        gt_mask = np.array(gt_mask_img.convert('L'))
        gt_percent = calculate_real_percentage(gt_mask)
        data.append((img, gt_mask, gt_percent))


    metrics = evaluate(local_process, data)
    pprint(metrics)
    visualize_metrics(metrics, save_path = 'metrics_540.png')
