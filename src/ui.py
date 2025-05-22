import os
import threading
import tkinter as tk
from openpyxl import Workbook
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Progressbar, Button, Label

from predict import process

class SkyOpennessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sky segmentation")
        self.geometry("500x400")
        self.images = []
        self.output_dir = ""

        # Кнопка выбора изображений
        btn1 = Button(self, text="Выбрать изображения", command=self.choose_images)
        btn1.pack(fill='x', padx=5, pady=(5,0))

        self.lbl_imgs = Label(self, text="Файлы не выбраны")
        self.lbl_imgs.pack(fill='x', padx=5, pady=(0,5))

        # Кнопка запуска
        btn3 = Button(self, text="Запустить обработку", command=self.start_processing)
        btn3.pack(fill='x', padx=5, pady=(10,5))

        # Прогресс-бар
        self.progress = Progressbar(self, orient='horizontal', length=580, mode='determinate')
        self.progress.pack(padx=5, pady=5)

        # Лог
        self.log = scrolledtext.ScrolledText(self, height=10)
        self.log.pack(fill='both', expand=True, padx=5, pady=5)

    def choose_images(self):
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png")]
        )
        if files:
            self.images = list(files)
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
        self.log_msg("Загрузка модели…")

        total = len(self.images)
        self.progress['maximum'] = total
        self.progress['value'] = 0

        excel_path = 'results.xlsx'
        wb = Workbook()
        ws = wb.active
        ws.append(['img_path', 'percentWith512Tile', 'percentWith1024Tile', 'percent'])

        for idx, img_path in enumerate(self.images, start=1):
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.log_msg(f"[{idx}/{total}] Обработка {base}…")

            percentWith512Tile = process(
                image_path=img_path,
                tile_size=512
            )

            percentWith1024Tile = process(
                image_path=img_path,
                tile_size=1024
            )

            percent = (percentWith512Tile + percentWith1024Tile) / 2

            ws.append([img_path, f"{percentWith512Tile:.2f}", f"{percentWith1024Tile:.2f}", f"{percent:.2f}"])

            self.log_msg(f"  → {percent:.2f}% неба")
            self.progress['value'] = idx

        # Сохраняем Excel файл
        wb.save(excel_path)
        self.log_msg(f"Результаты сохранены в {excel_path}")
        messagebox.showinfo("Готово", "Все изображения обработаны!")
        self.log_msg("Завершено.")

if __name__ == "__main__":
    SkyOpennessApp().mainloop()
