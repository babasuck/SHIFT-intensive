from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import DataLoader, Dataset


class SignDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transform

        labels = sorted(set(str(x).split('/')[-2] for x in paths))
        self.one_hot_encoding = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.paths[idx]))
        label = str(self.paths[idx]).split('/')[-2]

        # Add augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        image = cv2.resize(image, (200, 200))
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image).float(), torch.tensor(self.one_hot_encoding[label])


def get_sign_dataloader(
        path_train, path_val, batch_size, shuffle=True, num_workers=1,
    ):
    train_transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=(-.1, .1),
                           scale_limit=(-.2, .2),
                           rotate_limit=0,
                           p=1),
        A.OneOf([
            A.RandomToneCurve(scale=0.1, 
                              p=0.2),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), 
                                       contrast_limit=(-0.2, 0.2),
                                       p=0.8),
        ], p=0.3),                 
        A.GaussNoise(p=0.2),
        A.Normalize()])
    
    val_transform = A.Normalize()

    train_dataset = SignDataset(paths=[*Path(path_train).rglob('*.jpg')], transform=train_transform)
    val_dataset = SignDataset(paths=[*Path(path_val).rglob('*.jpg')], transform=val_transform)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def get_sign_test_dataloader(
        path_test, batch_size, num_workers=1,
    ):
    test_dataset = SignDataset(paths=[*Path(path_test).rglob('*.jpg')])

    loader_args = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers
    }
    return DataLoader(test_dataset, **loader_args)
