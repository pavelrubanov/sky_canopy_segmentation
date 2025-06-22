import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf


# --------------------------------------------------------------------------- #
#                           —---  С Л У Ж Е Б Н Ы Е  ---—                     #
# --------------------------------------------------------------------------- #
def resource_path(relative):
    """
    Возвращает корректный путь как в обычном запуске, так и из .exe.
    При сборке PyInstaller помещает все ресурсы во временную папку,
    путь к ней хранится в sys._MEIPASS.
    """
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, relative)


# ---------- utils ----------------------------------------------------------
def make_starts(full: int, tile: int) -> list[int]:
    starts = list(range(0, full - tile, tile))
    starts.append(full - tile)  # гарантируем охват правого края
    return starts


def sliding_windows(h: int, w: int, tile: int):
    y_starts = make_starts(h, tile)
    x_starts = make_starts(w, tile)
    for y0 in y_starts:
        for x0 in x_starts:
            yield y0, y0 + tile, x0, x0 + tile


def resize_large_image(img, max_size=1080):
    """Масштабирует изображение так, чтобы хотя бы один из размеров был не более max_size."""
    width, height = img.size

    if width <= max_size or height <= max_size:
        return img

    if height <= width:
        if height <= max_size:
            return img
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
    else:
        if width <= max_size:
            return img
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))

    return img.resize((new_width, new_height), Image.LANCZOS)


# --------------------------------------------------------------------------- #
#                           —---  О С Н О В Н О Е  ---—                       #
# --------------------------------------------------------------------------- #
def predict_mask(
        model: tf.keras.Model,
        img: np.ndarray,
        tile_size: int = 512,
        resize_to: int = 256,
) -> np.ndarray:
    """
    Вернёт вероятностную маску (float32, 0‒1) той же формы, что и *img*.
    """
    h, w = img.shape[:2]

    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_cnt = np.zeros((h, w), dtype=np.float32)

    for y0, y1, x0, x1 in sliding_windows(h, w, tile_size):
        tile = img[y0:y1, x0:x1, :]

        # --- downscale to 256×256 and model inference ----------------------
        tile_small = np.array(Image.fromarray(tile).resize((resize_to, resize_to), Image.LANCZOS))
        tile_small = tile_small.astype(np.float32) / 255.0
        tile_small = np.expand_dims(tile_small, 0)  # BCHW

        pred_small = model.predict(tile_small, verbose=0)[0, ...]  # (256, 256, 1) or (...,3)
        if pred_small.ndim == 3:
            pred_small = pred_small[..., 0]  # take single class

        # --- upsample back --------------------------------------
        pred_big = np.array(
            Image.fromarray(pred_small).resize((tile_size, tile_size), Image.LANCZOS),
            dtype=np.float32,
        )

        # --- insert into global canvas ------------------------------------
        prob_sum[y0:y1, x0:x1] += pred_big
        prob_cnt[y0:y1, x0:x1] += 1.0

    mask = prob_sum / prob_cnt
    return mask

def process(image_path, tile_size, model_path='./imageseg_canopy_model.hdf5', save=True):
    # ---------- load image --------------------------------------------------
    img = Image.open(image_path).convert("RGB")
    img = resize_large_image(img, max_size=1080)  # Масштабируем изображение
    img_np = np.array(img)

    # ---------- load model & predict ---------------------------------------
    relative_model_path = resource_path(model_path)
    model = tf.keras.models.load_model(relative_model_path, compile=False)

    mask = predict_mask(model, img_np, tile_size=tile_size)
    mask_vis = (np.clip(mask, 0, 1) * 255).round().astype(np.uint8)

    # ---------- save --------------------------------------------------------
    if save:
        out_path = image_path + f'_mask.png'
        Image.fromarray(mask_vis).save(out_path)

    return (np.mean(mask) * 100, mask)
