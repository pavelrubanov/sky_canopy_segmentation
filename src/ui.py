import os
import re
import threading
import tkinter as tk
from openpyxl import Workbook
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Progressbar, Button, Label, Checkbutton

from predict import process


class SkyOpennessApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sky segmentation")
        self.geometry("600x500")

        # запрет изменения размеров окна
        self.resizable(False, False)

        self.images: list[str] = []
        self.output_dir: str | None = None

        # --- Дополнительные переменные интерфейса ---
        self.save_masks_var = tk.BooleanVar(value=False)

        # Кнопка выбора изображений
        btn1 = Button(self, text="Выбрать изображения", command=self.choose_images)
        btn1.pack(fill='x', padx=5, pady=(5, 0))

        self.lbl_imgs = Label(self, text="Файлы не выбраны")
        self.lbl_imgs.pack(fill='x', padx=5, pady=(0, 5))

        # ----- Чекбокс сохранения ------
        opt_frame = tk.Frame(self)
        opt_frame.pack(fill='x', padx=5, pady=(0, 5))

        chk1 = Checkbutton(
            opt_frame,
            text="Сохранять маски",
            variable=self.save_masks_var
        )
        chk1.pack(side='left')
        # -----------------------------------------------------

        # Кнопка запуска
        btn3 = Button(self, text="Запустить обработку", command=self.start_processing)
        btn3.pack(fill='x', padx=5, pady=(10, 5))

        # Прогресс-бар
        self.progress = Progressbar(self, orient='horizontal', length=580, mode='determinate')
        self.progress.pack(padx=5, pady=5)

        # Лог
        self.log = scrolledtext.ScrolledText(self, height=10)
        self.log.pack(fill='both', expand=True, padx=5, pady=5)

    def choose_images(self):
        files = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.heic *.heif")])
        if files:
            self.images = list(files)
            # каталог, где лежит первый выбранный файл
            self.output_dir = os.path.dirname(self.images[0])
            self.lbl_imgs.config(text=f"{len(files)} файла(ов) выбрано")

    def log_msg(self, msg):
        self.log.insert('end', msg + "\n")
        self.log.see('end')

    def start_processing(self):
        if not self.images:
            messagebox.showerror("Ошибка", "Сначала выберите изображения.")
            return

        threading.Thread(target=self.process_all, daemon=True).start()

    def process_all(self):
        if not self.output_dir:
            messagebox.showerror("Ошибка", "Не удалось определить папку для сохранения результатов.")
            return

        self.log_msg("Загрузка модели…")

        total = len(self.images)
        self.progress['maximum'] = total
        self.progress['value'] = 0

        # ------------------------------------------------------------------
        # формируем уникальное имя вида result(N).xlsx  (0 ≤ N, целое)
        pattern = re.compile(r"result\((\d+)\)\.xlsx$")
        indices = [
            int(m.group(1))
            for name in os.listdir(self.output_dir)
            if (m := pattern.match(name))
        ]
        next_idx = max(indices) + 1 if indices else 0
        excel_filename = f"result({next_idx}).xlsx"
        excel_path = os.path.join(self.output_dir, excel_filename)
        # ------------------------------------------------------------------

        wb = Workbook()
        ws = wb.active
        ws.append(['img_name', 'percent'])

        save_masks = self.save_masks_var.get()
        tile_size = 810

        for idx, img_path in enumerate(self.images, start=1):
            img_name = os.path.basename(img_path)
            self.log_msg(f"[{idx}/{total}] Обработка {img_name}…")

            percent, _ = process(
                image_path=img_path,
                tile_size=tile_size,
                save=save_masks,
            )

            ws.append([img_name, f"{percent:.2f}"])

            self.log_msg(f"  → {percent:.2f}% неба")
            self.progress['value'] = idx

        wb.save(excel_path)
        self.log_msg(f"Результаты сохранены в {excel_path}")
        messagebox.showinfo("Готово", "Все изображения обработаны!")
        self.log_msg("Завершено.")


if __name__ == "__main__":
    SkyOpennessApp().mainloop()