import os
import json
import operator
import random
import shutil
import glob
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm


def augment_image_flip(folder_path):
    if not os.path.exists(folder_path):
        print(f"No folder in {folder_path}")
        return

    image_files = [file for file in os.listdir(folder_path) if file.endswith((".jpg", ".jpeg", ".png"))]

    total_files = len(image_files)

    if total_files == 0:
        print("No images in the folder")
        return

    for file in tqdm(image_files, desc="Processing", total=total_files):
        image_path = os.path.join(folder_path, file)
        image = Image.open(image_path)
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        new_file_name = "_" + file
        flipped_image_path = os.path.join(folder_path, new_file_name)
        flipped_image.save(flipped_image_path)
        image.close()

    print("Images saved with prefix \"_\".")


# File path to your annotations JSON
def conv_annot(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    progress_bar = tqdm(total=len(data['images']), desc="Processing images")

    for image in data['images']:
        id_s, file_name, w, h = (image['id'], image['file_name'], image['width'], image['height'])
        txt = file_name.split('.')[0] + '.txt'
        txt = 'new_screen/' + txt

        with open(txt, "w", encoding='utf-8') as folder:
            for annotations in data['annotations']:
                if annotations['image_id'] == id_s:
                    folder.write(str(annotations['category_id'] - 1) + ' ')
                    for idx, g in enumerate(annotations['segmentation'][0]):
                        if idx % 2 == 0:
                            folder.write(str(operator.truediv(g, w)) + ' ')
                        else:
                            folder.write(str(operator.truediv(g, h)) + ' ')
                    folder.write('\n')

        progress_bar.update(1)

    progress_bar.close()


# File path to your annotations JSON
def conv_annot_flip(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    progress_bar = tqdm(total=len(data['images']), desc="Processing images")

    for image in data['images']:
        id_s, file_name, w, h = (image['id'], image['file_name'], image['width'], image['height'])
        file_name = '_' + file_name
        txt = file_name.split('.')[0] + '.txt'
        txt = 'new_screen/' + txt

        with open(txt, "w", encoding='utf-8') as folder:
            for annotations in data['annotations']:
                if annotations['image_id'] == id_s:
                    folder.write(str(annotations['category_id'] - 1) + ' ')
                    for idx, g in enumerate(annotations['segmentation'][0]):
                        if idx % 2 == 0:
                            folder.write(str(operator.truediv((1920 - g), w)) + ' ')
                        else:
                            folder.write(str(operator.truediv(g, h)) + ' ')
                    folder.write('\n')

        progress_bar.update(1)

    progress_bar.close()


# Path to your folder with your images and annotations(this is one same folder)
def split_dataset(source_folder, train_ratio, test_ratio, valid_ratio):
    # Создаем папки train, test, valid
    os.makedirs('train', exist_ok=True)
    os.makedirs('valid', exist_ok=True)
    os.makedirs('test', exist_ok=True)

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
        target_path_img = os.path.join('train', base_name)
        target_path_ann = os.path.join('train', base_name.replace('.jpg', '.txt'))
        shutil.copy(file_name, target_path_img)
        shutil.copy(file_name.replace('.jpg', '.txt'), target_path_ann)

    test_files = img_files[num_train:num_train + num_test]
    for file_name in test_files:
        base_name = os.path.basename(file_name)
        target_path_img = os.path.join('test', base_name)
        target_path_ann = os.path.join('test', base_name.replace('.jpg', '.txt'))
        shutil.copy(file_name, target_path_img)
        shutil.copy(file_name.replace('.jpg', '.txt'), target_path_ann)

    valid_files = img_files[num_train + num_test:]
    for file_name in valid_files:
        base_name = os.path.basename(file_name)
        target_path_img = os.path.join('valid', base_name)
        target_path_ann = os.path.join('valid', base_name.replace('.jpg', '.txt'))
        shutil.copy(file_name, target_path_img)
        shutil.copy(file_name.replace('.jpg', '.txt'), target_path_ann)


def train_Yolo(path_yaml):
    # Load the model
    model = YOLO('yolov8l-seg.pt')
    results = model.train(data=path_yaml, epochs=120, batch=16, device=0, imgsz=1280)