from os.path import join, exists, basename, dirname

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def get_transforms(img_size=(224, 224), is_train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ])
        return train_transform
    else:
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
        ])
        return test_transform


# this function is responsible to handle data imbalance
def get_sampler(target_dataset: "ImageFolder"):
    u_values, u_count = np.unique(target_dataset.targets, return_counts=True)
    weights = 1 / u_count
    sample_weights = [0] * len(target_dataset)
    for idx, (_, label) in enumerate(tqdm(target_dataset)):
        sample_weights[idx] = weights[label]
    return WeightedRandomSampler(sample_weights, num_samples=len(target_dataset), replacement=True)


def get_dataloader(path, only_dataset=True, batch_size=32, **kwargs):
    """
     This function creates dataloader with random weighted sampler
    :param batch_size:
    :param path:  This path should contain 3 folder train test validation
    :return: Dataset, Dataloaders
    """
    folders = ["train", 'test', 'validation']
    data_paths = [join(path, f) for f in folders]

    # check if they exists
    for data_path in data_paths:
        assert exists(data_path), f"{basename(data_path)} does not exists in {dirname(data_path)}"
    loaders = dict()
    data_sets = dict()
    for folder, data_path in zip(folders, data_paths):
        data_set = ImageFolder(data_path, get_transforms(is_train=folder == "train"))
        if only_dataset:
            data_sets[folder] = data_set
        else:
            if folder == 'train':
                sampler = get_sampler(data_set)
            else:
                sampler = None
            loaders[folder] = DataLoader(data_set, batch_size, sampler=sampler, **kwargs)
            data_sets[folder] = data_set
    if only_dataset:
        return data_sets
    return data_sets, loaders


