import os
from ultralytics import YOLO
from pathlib import Path


model = YOLO("run6.pt")
# model = YOLO("run3.pt")
# model = YOLO("run2.pt")

input_folder = Path("/home/moo/PycharmProjects/Train_Yolo/test_folder")
output_folder = Path("run6_predict")
output_folder.mkdir(parents=True, exist_ok=True)

# Поддерживаемые форматы изображений
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# Обработка всех изображений в папке
for image_file in input_folder.iterdir():
    if image_file.suffix.lower() in image_extensions:
        results = model(str(image_file), conf=0.3)
        for i, result in enumerate(results):
            output_filename = output_folder / f"{image_file.stem}_result_{i}.jpg"
            result.save(filename=str(output_filename))

print("Обработка завершена. Результаты сохранены в папке run5_predict.")
