"""
Сборка проекта Sky segmentation в .app-пакет для macOS с помощью PyInstaller.

Подготовка:
  1. Откройте терминал и перейдите в каталог src.
  2. Активируйте виртуальную среду:     source venv/bin/activate
  3. Установите PyInstaller (если нужно): pip install pyinstaller
Запуск:
    (venv) $ python build_app_mac.py
После завершения в папке dist появится SkySegmentation.app.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


if sys.platform != "darwin":
    sys.exit("Скрипт предназначен только для macOS.")

PROJECT_ROOT = Path(__file__).resolve().parent
ENTRY_POINT  = PROJECT_ROOT / "src/ui.py"
MODEL_FILE   = PROJECT_ROOT / "src/imageseg_canopy_model.hdf5"
APP_NAME     = "SkySegmentation"


def clean_previous_builds() -> None:
    for artefact in ("build", "dist", f"{APP_NAME}.spec"):
        path = PROJECT_ROOT / artefact
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink(missing_ok=True)


def build_mac_app() -> None:
    add_data = f"{MODEL_FILE}:."

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", APP_NAME,
        "--add-data", add_data,
        # скрытые импорты для TensorFlow/Keras
        "--hidden-import", "keras.api",
        "--hidden-import", "keras.utils",
        str(ENTRY_POINT)
    ]

    subprocess.check_call(cmd, cwd=PROJECT_ROOT)

    final_app = PROJECT_ROOT / "dist" / f"{APP_NAME}.app"


if __name__ == "__main__":
    clean_previous_builds()
    build_mac_app()