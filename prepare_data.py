import os
from os.path import join, isdir, dirname, exists, basename
from shutil import copyfile

from sklearn.model_selection import train_test_split

a = {'Basmati': 15000, 'Arborio': 9000, 'Karacadag': 7000, 'Jasmine': 4000, 'Ipsala': 2000}


def prepare_data(path, destination_path: str = None):
    import random
    EXTENSIONS = ('jpg', 'jpeg', 'bmp', 'tiff', 'png', 'eps')
    if destination_path is None:
        destination_path = os.path.join(dirname(path), "prepare_data")
    folders = [i for i in os.listdir(path) if isdir(join(path, i))]
    for n, f in enumerate(folders):
        image_names = [i for i in os.listdir(join(path, f)) if i.endswith(EXTENSIONS)]
        image_names = random.choices(image_names, k=a[f])
        for name in image_names:
            if not exists(join(destination_path, f)):
                os.makedirs(join(destination_path, f))
            copyfile(join(path, f, name), join(destination_path, f, name))


def split_dataset(path, destination_path, test_ratio=0.3, val_ratio=0.1):
    train_ratio = 1 - test_ratio - val_ratio
    assert train_ratio > 0, "train ratio cannot be zero"
    splits = ["train", 'test', 'validation']
    data = []
    labels = []
    for n, i in enumerate(os.listdir(path)):
        if isdir(join(path, i)):
            data.extend([join(path, i, name) for name in os.listdir(join(path, i))])
            labels.extend([n] * len(os.listdir(join(path, i))))
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_ratio, stratify=labels)
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=val_ratio, stratify=labels)

    data = [x_train, x_test, x_val]
    for folder_name, split_data in zip(splits, data):
        for d in split_data:
            if not exists(join(destination_path, folder_name, basename(dirname(d)))):
                os.makedirs(join(destination_path, folder_name, basename(dirname(d))))
            copyfile(d, join(destination_path, folder_name, basename(dirname(d)), basename(d)))


# prepare_data('/home/sarvesh/Documents/repositories/sarvesh-personal/grain-classification/Rice_Image_Dataset/data')

path = '/home/sarvesh/Documents/repositories/sarvesh-personal/grain-classification/Rice_Image_Dataset/data_splits/test'
split_dataset(path, '/home/sarvesh/Documents/repositories/sarvesh-personal/grain-classification/Rice_Image_Dataset/new_split')