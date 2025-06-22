from pathlib import Path
import numpy as np
from PIL import Image
from openpyxl import Workbook
from src.predict import process
from utils import calculate_real_percentage

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
    data_dir = Path('data') # Папка с изображениями в формате jpg
    masks_dir = Path('masks') # Папка с масками в формате png

    # Заголовок будущей таблицы
    excel_rows = [(
        'Image #',
        'Pred 256, %',
        'Pred 512, %',
        'Pred 1024, %',
        'Real min, %',
        'Real max, %'
    )]

    for image_path in sorted(data_dir.glob('*.jpg')):
        image_name = image_path.stem  # имя файла без расширения
        mask_path = masks_dir / f'{image_name}.png'  # маска с тем же именем

        if not image_path.exists() or not mask_path.exists():
            print(f"Пропуск image_path: файлы не найдены")
            continue

        mask = np.array(Image.open(mask_path).convert('L'))

        model_path = '../src/imageseg_canopy_model.hdf5'

        percentage_256, mask_256 = process(str(image_path), model_path=model_path, tile_size=256, save=True)
        percentage_512, mask_512 = process(str(image_path), model_path=model_path, tile_size=512, save=True)
        percentage_1024, mask_1024 = process(str(image_path), model_path=model_path, tile_size=1024, save=True)

        real_percent_min, real_percent_max = calculate_real_percentage(mask)

        excel_rows.append(
            (
                image_name,
                round(percentage_256, 4),
                round(percentage_512, 4),
                round(percentage_1024, 4),
                round(real_percent_min, 4),
                round(real_percent_max, 4)
            )
        )

    # Сохраняем результаты после обработки всех изображений
    if len(excel_rows) > 1:  # если есть хотя бы одна строка с данными
        save_to_excel(excel_rows, 'results1.xlsx')
    else:
        print('Нет данных для сохранения.')


if __name__ == '__main__':
    main()