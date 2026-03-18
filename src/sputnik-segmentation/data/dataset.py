import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from albumentations import BasicTransform
import pytorch_lightning as pl


class SputnikSegDataset(Dataset):
    def __init__(self,
                 image_paths,
                 mask_paths,
                 num_classes,
                 unique_classes,
                 transforms: BasicTransform = None
                 ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes
        self.unique_classes = unique_classes
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        if self.num_classes > 1:
            mask = self.mask_to_one_hot(mask)
        else:
            mask = np.where(mask > 0, 1, 0)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.num_classes == 1:
            mask = mask.unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.image_paths)

    def mask_to_one_hot(self, mask: pd.DataFrame):
        return (mask[:, :, None] == self.unique_classes).astype(np.uint8)


class SputnikSegDataloader(pl.LightningDataModule):
    def __init__(self,
                 image_paths,
                 mask_paths,
                 num_classes,
                 unique_classes,
                 batch_size,
                 train_transforms:BasicTransform,
                 val_transforms:BasicTransform,
                 test_transforms:BasicTransform):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.classes = num_classes
        self.unique_classes = unique_classes
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def setup(self, stage=None):
        # Создаем кастомные датасеты для обучения, валидации и тестирования
        self.train_dataset = SputnikSegDataset(image_paths=self.image_paths,
                                              mask_paths=self.mask_paths,
                                              num_classes=self.classes,
                                              unique_classes=self.unique_classes,
                                              transforms=self.train_transforms)

        self.val_dataset = SputnikSegDataset(image_paths=self.image_paths,
                                            mask_paths=self.mask_paths,
                                            num_classes=self.classes,
                                            unique_classes=self.unique_classes,
                                            transforms=self.val_transforms)

        self.test_dataset = SputnikSegDataset(image_paths=self.image_paths,
                                             mask_paths=self.mask_paths,
                                             num_classes=self.classes,
                                             unique_classes=self.unique_classes,
                                             transforms=self.test_transforms)

        # Перемешиваем данные случайным образом
        indices = torch.randperm(len(self.train_dataset))

        # Деля данные на тренировочные, валидационные и тестовые
        train_set_size = int(len(self.train_dataset) * 0.80)
        valid_set_size = int(len(self.train_dataset) * 0.10)
        test_set_size = len(self.train_dataset) - train_set_size - valid_set_size

        # Создаем подмножества данных для тренировки, валидации и тестирования
        self.train_data = torch.utils.data.Subset(self.train_dataset, indices[:train_set_size])
        self.validation_data = torch.utils.data.Subset(self.val_dataset,
                                                       indices[train_set_size:(train_set_size + valid_set_size)])
        self.test_data = torch.utils.data.Subset(self.test_dataset, indices[-test_set_size:])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=False, num_workers=4,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=4)