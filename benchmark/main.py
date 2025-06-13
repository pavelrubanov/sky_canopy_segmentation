from pathlib import Path
import numpy as np
from PIL import Image
from src.predict import process

def calculate_real_percentage(mask_path):
    mask = np.array(Image.open(mask_path).convert('L'))
    white_pixels = np.sum(mask > 127)  # считаем пиксели светлее 127 как белые
    total_pixels = mask.size
    return (white_pixels / total_pixels) * 100

def main():
    test_data_dir = Path('low_test_data')
    
    for i in range(1, 10):
        image_path = test_data_dir / f'{i}.jpg'
        mask_path = test_data_dir / f'{i}_mask.png'
        
        if not image_path.exists() or not mask_path.exists():
            print(f"Пропуск {i}: файлы не найдены")
            continue
            
        # Получаем предсказание модели
        model_path = '../src/imageseg_canopy_model.hdf5'
        percentage_with_512_tile = process(str(image_path), model_path=model_path,  tile_size=512, save=True)
        percentage_with_512_tile_with_threshold = process(str(image_path), model_path=model_path,  tile_size=512, save=True, threshold=0.5)

        percentage_with_1024_tile = process(str(image_path), model_path=model_path, tile_size=1024, save=True)
        percentage_with_1024_tile_with_threshold = process(str(image_path), model_path=model_path, tile_size=1024, save=True, threshold=0.5)

        average_percent1 = (percentage_with_512_tile + percentage_with_1024_tile) / 2
        average_percent2 = (percentage_with_512_tile_with_threshold + percentage_with_1024_tile_with_threshold) / 2
        
        # Считаем реальный процент
        real_percentage = calculate_real_percentage(mask_path)
        
        print(f"Изображение {i}:")
        print(f"  Предсказано 512 tile: {percentage_with_512_tile:.2f}%, 1024 tile: {percentage_with_1024_tile:.2f}%")
        print(f"  Предсказано avarage: {average_percent1:.2f}%")
        print(f"  Предсказано 512 tile: {percentage_with_512_tile_with_threshold:.2f}%, 1024 tile: {percentage_with_1024_tile_with_threshold:.2f}%")
        print(f"  Предсказано avarage: {average_percent2:.2f}%")
        print(f"  Реально: {real_percentage:.2f}%")
        print()

if __name__ == '__main__':
    main()
