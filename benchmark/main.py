from pathlib import Path
import numpy as np
from PIL import Image
from openpyxl import Workbook
from src.predict import process, resize_large_image

def calculate_real_percentage(mask):
    total_pixels = mask.size

    white_pixels = np.sum(mask > 254)
    percent_min = (white_pixels / total_pixels) * 100

    all_pixels_sum = np.sum(mask)
    percent_max = (all_pixels_sum / (total_pixels * 255)) * 100

    return percent_min, percent_max


def save_to_excel(rows, filename='results.xlsx'):
    """
    Сохраняет список строк (list[tuple]) в Excel-файл.
    Первая строка должна содержать заголовки.
    """
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    wb.save(filename)
    print(f'Данные сохранены в файл {filename}')


def main():
    test_data_dir = Path('test_data')

    # Заголовок будущей таблицы
    excel_rows = [(
        'Image #',
        'Pred 512, %',
        'Pred 1024, %',
        'Pred Avg, %',
        'Pred 512 thr, %',
        'Pred 1024 thr, %',
        'Pred Avg thr, %',
        'Real min, %',
        'Real max, %',
        'Resized real min, %',
        'Resized real max, %'
    )]

    for i in range(1, 10):
        image_path = test_data_dir / f'{i}.jpg'
        mask_path = test_data_dir / f'{i}_mask.png'

        if not image_path.exists() or not mask_path.exists():
            print(f"Пропуск {i}: файлы не найдены")
            continue

        mask = np.array(Image.open(mask_path).convert('L'))
        resized_mask = np.array(resize_large_image(Image.open(mask_path), max_size=1024).convert('L'))

        model_path = '../src/imageseg_canopy_model.hdf5'
        percentage_with_512_tile = process(
            str(image_path), model_path=model_path, tile_size=512, save=True
        )
        percentage_with_512_tile_with_threshold = process(
            str(image_path),
            model_path=model_path,
            tile_size=512,
            save=True,
            threshold=0.5,
        )
        percentage_with_1024_tile = process(
            str(image_path), model_path=model_path, tile_size=1024, save=True
        )
        percentage_with_1024_tile_with_threshold = process(
            str(image_path),
            model_path=model_path,
            tile_size=1024,
            save=True,
            threshold=0.5,
        )

        average_percent1 = (percentage_with_512_tile + percentage_with_1024_tile) / 2
        average_percent2 = (percentage_with_512_tile_with_threshold+ percentage_with_1024_tile_with_threshold) / 2

        real_percent_min, real_percent_max = calculate_real_percentage(mask)
        resized_real_percent_min, resized_real_percent_max = calculate_real_percentage(resized_mask)

        # Добавляем строку для Excel
        excel_rows.append(
            (
                i,
                round(percentage_with_512_tile, 4),
                round(percentage_with_1024_tile, 4),
                round(average_percent1, 4),
                round(percentage_with_512_tile_with_threshold, 4),
                round(percentage_with_1024_tile_with_threshold, 4),
                round(average_percent2, 4),
                round(real_percent_min, 4),
                round(real_percent_max, 4),
                round(resized_real_percent_min, 4),
                round(resized_real_percent_max, 4),
            )
        )

    # Сохраняем результаты после обработки всех изображений
    if len(excel_rows) > 1:  # если есть хотя бы одна строка с данными
        save_to_excel(excel_rows, 'results1.xlsx')
    else:
        print('Нет данных для сохранения.')


if __name__ == '__main__':
    main()