import json
import shutil
import os
from tqdm import tqdm
import cv2
import random
import glob
from PIL import Image


def conv_annot(file_path, work_path):
    """
    :param work_path: Рабочая директория с изображениями
    :param file_path: путь к JSOn
    :return: сохраняет аннотации в формате .txt
    """
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    progress_bar = tqdm(total=len(data['images']), desc="Processing images")

    for image in data['images']:
        id_s, file_name, w, h = (image['id'], image['file_name'], image['width'], image['height'])

        txt = file_name.rsplit('.', 1)[0] + '.txt'
        txt = os.path.join(f'{work_path}', txt)
        print(txt)
        os.makedirs(os.path.dirname(txt), exist_ok=True)

        with open(txt, "w", encoding='utf-8') as folder:
            for annotations in data['annotations']:
                if annotations['image_id'] == id_s:
                    folder.write(str(annotations['category_id'] - 1) + ' ')
                    for idx, g in enumerate(annotations['segmentation'][0]):
                        if idx % 2 == 0:
                            folder.write(f"{g / w:.6f} ")
                        else:
                            folder.write(f"{g / h:.6f} ")
                    folder.write('\n')

        progress_bar.update(1)

    progress_bar.close()



def conv_annot_flip(file_path, work_path):
    """
    :param work_path: Рабочая директория с изображениями
    :param file_path: путь к JSOn
    :return: сохраняет флипнутые аннотации в формате .txt
    """
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    progress_bar = tqdm(total=len(data['images']), desc="Processing images")

    for image in data['images']:
        id_s, file_name, w, h = (image['id'], image['file_name'], image['width'], image['height'])

        # Add '_'
        txt = '_' + file_name.rsplit('.', 1)[0] + '.txt'
        txt = os.path.join(f'{work_path}', txt)

        os.makedirs(os.path.dirname(txt), exist_ok=True)

        with open(txt, "w", encoding='utf-8') as folder:
            for annotations in data['annotations']:
                if annotations['image_id'] == id_s:
                    folder.write(f"{annotations['category_id'] - 1} ")
                    segmentation = annotations['segmentation'][0]
                    for idx in range(0, len(segmentation), 2):
                        x, y = segmentation[idx], segmentation[idx + 1]
                        # Flip on X, Y = const
                        flipped_x = (w - x) / w
                        flipped_y = y / h
                        folder.write(f"{flipped_x:.6f} {flipped_y:.6f} ")
                    folder.write('\n')

        progress_bar.update(1)

    progress_bar.close()



def augment_image_flip(input_folder_path):
    """
    :param input_folder_path: Рабочая директория с изображениями
    :return: Сохраняет флипнутые изображения
    """
    if not os.path.exists(input_folder_path):
        print(f"No folder in {input_folder_path}")
        return

    if not os.path.exists(input_folder_path):
        os.makedirs(input_folder_path)
        print(f"Created output folder: {input_folder_path}")

    image_files = [file for file in os.listdir(input_folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]

    total_files = len(image_files)

    if total_files == 0:
        print("No images in the input folder")
        return

    for file in tqdm(image_files, desc="Processing", total=total_files):
        image_path = os.path.join(input_folder_path, file)
        image = Image.open(image_path)
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        new_file_name = "_" + file
        flipped_image_path = os.path.join(input_folder_path, new_file_name)
        flipped_image.save(flipped_image_path)
        image.close()

    print(f"Flipped images saved with prefix \"_\" in {input_folder_path}.")


def remove_trailing_space(folder_path):
    """
    Функция для удаления пробелом в конце аннотаций
    :param folder_path: Рабочая директория с изображениями
    :return:
    """
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    progress_bar = tqdm(total=len(txt_files), desc="Processing files")

    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        modified_lines = [line.rstrip() + '\n' for line in lines]

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)
        progress_bar.update(1)

    progress_bar.close()

    print(f"Обработано {len(txt_files)} файлов.")



def augment_image(image):
    """
    Функция для аугментации
    :param image: Кадр
    :return: Аугментированный кадр
    """
    augmentations = [
        lambda img: cv2.convertScaleAbs(img, alpha=random.uniform(0.5, 1.5), beta=random.randint(-50, 50)),
        lambda img: cv2.GaussianBlur(img, (5, 5), random.uniform(0, 3)),
        lambda img: cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 30), -4, 128),
    ]

    aug = random.choice(augmentations)
    return aug(image)


def process_images_and_annotations(folder):
    """
    Применение аугментации
    :param folder: Рабочая директория с изображениями
    :return:
    """
    # # Создаем выходные папки, если они не существуют
    # os.makedirs(output_image_folder, exist_ok=True)
    # os.makedirs(output_annotation_folder, exist_ok=True)

    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    progress_bar = tqdm(total=len(image_files), desc="Processing images and annotations")

    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        image = cv2.imread(image_path)
        augmented_image = augment_image(image)
        base_name, ext = os.path.splitext(image_file)
        new_image_name = f"{base_name}_aug{ext}"
        new_image_path = os.path.join(folder, new_image_name)

        cv2.imwrite(new_image_path, augmented_image)

        annotation_file = f"{base_name}.txt"
        annotation_path = os.path.join(folder, annotation_file)

        if os.path.exists(annotation_path):
            new_annotation_name = f"{base_name}_aug.txt"
            new_annotation_path = os.path.join(folder, new_annotation_name)
            shutil.copy(annotation_path, new_annotation_path)

        progress_bar.update(1)

    progress_bar.close()
    print(f"Обработано {len(image_files)} изображений и аннотаций.")


def split_dataset(source_folder, train_ratio, test_ratio, valid_ratio):
    """
    Функция дробления на трейн тест валид
    :param source_folder:
    :param train_ratio:
    :param test_ratio:
    :param valid_ratio:
    :return:
    """
    # Создаем основную папку train_dir
    train_folder = 'train_dirrrrr'
    os.makedirs(train_folder, exist_ok=True)

    # Создаем папки train, test, valid внутри train_dir
    os.makedirs(os.path.join(train_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'test'), exist_ok=True)

    # Get the images with their annotations
    img_files = glob.glob(os.path.join(source_folder, '*.jpg'))
    ann_files = glob.glob(os.path.join(source_folder, '*.txt'))
    img_files.sort()
    ann_files.sort()

    # Mix the files
    random.seed(123)
    random.shuffle(img_files)
    random.seed(123)
    random.shuffle(ann_files)

    # Count
    num_files = len(img_files)
    num_train = round(num_files * train_ratio / 100)
    num_test = round(num_files * test_ratio / 100)
    num_valid = round(num_files * valid_ratio / 100)

    # Copy files to the three folders
    train_files = img_files[:num_train]
    for file_name in train_files:
        base_name = os.path.basename(file_name)
        target_path_img = os.path.join(train_folder, 'train', base_name)
        target_path_ann = os.path.join(train_folder, 'train', base_name.replace('.jpg', '.txt'))
        shutil.copy(file_name, target_path_img)
        shutil.copy(file_name.replace('.jpg', '.txt'), target_path_ann)

    test_files = img_files[num_train:num_train + num_test]
    for file_name in test_files:
        base_name = os.path.basename(file_name)
        target_path_img = os.path.join(train_folder, 'test', base_name)
        target_path_ann = os.path.join(train_folder, 'test', base_name.replace('.jpg', '.txt'))
        shutil.copy(file_name, target_path_img)
        shutil.copy(file_name.replace('.jpg', '.txt'), target_path_ann)

    valid_files = img_files[num_train + num_test:]
    for file_name in valid_files:
        base_name = os.path.basename(file_name)
        target_path_img = os.path.join(train_folder, 'valid', base_name)
        target_path_ann = os.path.join(train_folder, 'valid', base_name.replace('.jpg', '.txt'))
        shutil.copy(file_name, target_path_img)
        shutil.copy(file_name.replace('.jpg', '.txt'), target_path_ann)


def process_annotations(directory):
    """
    Функция перевода всех аннотаций к одному классу
    :param directory: Рабочая директория
    :return:
    """
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    progress_bar = tqdm(total=len(txt_files), desc="Processing files")

    for filename in txt_files:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        file_changed = False
        new_lines = []
        for line in lines:
            parts = line.split()
            if parts and parts[0] != '0':
                parts[0] = '0'
                new_line = ' '.join(parts) + '\n'
                file_changed = True
            else:
                new_line = line
            new_lines.append(new_line)
        if file_changed:
            with open(file_path, 'w') as file:
                file.writelines(new_lines)
        progress_bar.update(1)
    progress_bar.close()

    print(f"Обработано {len(txt_files)} файлов.")



def main(path_to_images, annotation_path, train_ratio, test_ratio, valid_ratio):
    """
    Функция запуска
    :param path_to_images: Путь к папке с изображениями
    :param annotation_path: Пусть к файлу аннотаций JSON
    :param train_ratio: Доля трейна
    :param test_ratio: Доля теста
    :param valid_ratio: Доля валида
    :return: Папку train_dirrrrr, по которой можно делать трейн
    """
    # Флип изображения
    augment_image_flip(path_to_images)
    # Распаковка аннотаций из JSON
    conv_annot(annotation_path, path_to_images)
    # Распаковка аннотаций из JSON + флип
    conv_annot_flip(annotation_path, path_to_images)
    # Убрать пробелы по всех аннотациях
    remove_trailing_space(path_to_images)
    # Дополнительная аугментация по параметрам фото
    process_images_and_annotations(path_to_images)
    # Приведение всех аннотаций к одному классу, если нужно
    process_annotations(path_to_images)
    # Сплит датасета
    split_dataset(path_to_images, train_ratio, test_ratio, valid_ratio)



if __name__ == "__main__":
    folder_with_images = '/home/moo/PycharmProjects/Train_Yolo/images'
    path_to_annotations = '/home/moo/PycharmProjects/Train_Yolo/result.json'
    main(folder_with_images, path_to_annotations, 75, 1, 24)
