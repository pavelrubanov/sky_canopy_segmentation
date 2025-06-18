# build_exe.py
"""
Собирает приложение Sky segmentation в один исполняемый файл.
Перед запуском убедитесь, что активирована виртуальная среда и установлен PyInstaller:
    (venv) > pip install pyinstaller
Запуск:
    (venv) > python build_exe.py
По завершении готовый exe появится в каталоге ./dist
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ENTRY_POINT   = PROJECT_ROOT / "ui.py"
MODEL_FILE    = PROJECT_ROOT / "imageseg_canopy_model.hdf5"
EXE_NAME      = "sky_segmentation"


def clean_previous_builds() -> None:
    """Удаляем artefacts предыдущих сборок."""
    for item in ("build", "dist", f"{EXE_NAME}.spec"):
        path = PROJECT_ROOT / item
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink(missing_ok=True)


def build_exe() -> None:
    """Формирует команду PyInstaller и запускает сборку."""
    # Формат --add-data для Windows: 'источник;назначение_внутри_архива'
    add_data = f"{MODEL_FILE};."

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--noconfirm",
        "--name", EXE_NAME,
        "--add-data", add_data,
        # скрытые импорты для корректной работы TensorFlow/Keras
        "--hidden-import", "keras.api",
        "--hidden-import", "keras.utils",
        str(ENTRY_POINT)
    ]

    print("Выполняется сборка:\n", " ".join(cmd), "\n")
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)
    print("\nСборка завершена. Файл находится в:", PROJECT_ROOT / "dist" / f"{EXE_NAME}.exe")


if __name__ == "__main__":
    clean_previous_builds()
    build_exe()