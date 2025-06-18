import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import os

class ImageMaskUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Настройка маски изображения")
        self.root.geometry("800x600")
        
        # Переменные
        self.original_mask = None
        self.current_image = None
        self.threshold = tk.IntVar(value=128)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Фрейм для кнопок
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # Кнопка выбора изображения
        ttk.Button(control_frame, text="Выбрать изображение", 
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        
        # Метка для отображения пути
        self.path_label = ttk.Label(control_frame, text="Файл не выбран")
        self.path_label.pack(side=tk.LEFT, padx=10)
        
        # Фрейм для ползунка
        slider_frame = ttk.Frame(self.root)
        slider_frame.pack(pady=10, fill=tk.X, padx=20)
        
        ttk.Label(slider_frame, text="Порог (0-255):").pack(side=tk.LEFT)
        
        # Ползунок
        self.slider = ttk.Scale(slider_frame, from_=0, to=255, 
                               variable=self.threshold, orient=tk.HORIZONTAL,
                               command=self.on_threshold_change)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Метка со значением
        self.value_label = ttk.Label(slider_frame, text="128")
        self.value_label.pack(side=tk.LEFT)
        
        # Фрейм для изображения
        self.image_frame = ttk.Frame(self.root, relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Canvas для отображения изображения
        self.canvas = tk.Canvas(self.image_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Информационная панель
        info_frame = ttk.Frame(self.root)
        info_frame.pack(pady=5, fill=tk.X, padx=20)
        
        self.info_label = ttk.Label(info_frame, 
                                   text="Выберите изображение для начала работы")
        self.info_label.pack()
        
    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Загружаем и преобразуем изображение в маску
                self.original_mask = np.array(Image.open(file_path).convert('L'))
                
                # Обновляем UI
                self.path_label.config(text=os.path.basename(file_path))
                self.update_image()
                
                # Обновляем информацию
                height, width = self.original_mask.shape
                self.info_label.config(
                    text=f"Размер: {width}x{height}, Порог: {self.threshold.get()}"
                )
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")
    
    def on_threshold_change(self, value):
        """Обработка изменения порога"""
        threshold_value = int(float(value))
        self.value_label.config(text=str(threshold_value))
        
        if self.original_mask is not None:
            self.update_image()
            
            # Обновляем информацию
            height, width = self.original_mask.shape
            visible_pixels = np.sum(self.original_mask > threshold_value)
            total_pixels = height * width
            percentage = (visible_pixels / total_pixels) * 100
            
            self.info_label.config(
                text=f"Размер: {width}x{height}, Порог: {threshold_value}, "
                     f"Видимых пикселей: {visible_pixels} ({percentage:.1f}%)"
            )
    
    def update_image(self):
        """Обновление отображаемого изображения"""
        if self.original_mask is None:
            return
        
        # Создаем маску на основе порога
        threshold_value = self.threshold.get()
        mask = self.original_mask > threshold_value
        
        # Создаем изображение для отображения
        display_image = np.zeros_like(self.original_mask)
        display_image[mask] = 255  # Белые пиксели для видимых областей
        
        # Преобразуем в PIL Image
        pil_image = Image.fromarray(display_image, mode='L')
        
        # Масштабируем изображение, если оно слишком большое
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_image.size
            
            # Вычисляем масштаб для помещения изображения в canvas
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y, 1.0)  # Не увеличиваем изображение
            
            if scale < 1.0:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.NEAREST)
        
        # Преобразуем в PhotoImage для tkinter
        self.current_image = ImageTk.PhotoImage(pil_image)
        
        # Очищаем canvas и отображаем новое изображение
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.current_image,
            anchor=tk.CENTER
        )
        
        # Обновляем область прокрутки
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

def main():
    root = tk.Tk()
    app = ImageMaskUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()