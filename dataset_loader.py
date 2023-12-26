import glob
import os
import random

import cv2
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from utils.common import seed_everything
seed_everything(seed=42)
cv2.setNumThreads(1)
import collections

from torch.utils.data.sampler import Sampler
import numpy as np

NUMPY_RANDOM_STATE = np.random.RandomState()
NUMPY_RANDOM = np.random

NUMPY_RANDOM_STATE = np.random.RandomState()
NUMPY_RANDOM = np.random


class CRC_Dataset(Dataset):
    """
    CRC_Dataset is a custom dataset class for loading and processing CRC images.
    It inherits from the PyTorch Dataset class and overrides the __init__, __len__, and __getitem__ methods.
    """

    def __init__(self, dataset_path, transform=None):
        """
        Initialize the CRC_Dataset with the given dataset_path and optional transform.
        Args:
            dataset_path (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self.datalist = []
        self.label_dict = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
        cls_paths = glob.glob('{}/*/'.format(dataset_path))

        with tqdm(enumerate(sorted(cls_paths)), disable=True) as t:
            for wj, cls_path in t:
                cls_id = str(os.path.split(os.path.dirname(cls_path))[-1])
                patch_pths = glob.glob('{}/*.tif'.format(cls_path))
                for pth in patch_pths:
                    self.datalist.append((pth, cls_id))
        self.targets = [self.label_dict[img_data[1]] for img_data in self.datalist]

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.datalist)

    def __getitem__(self, idx):
        """
        Get the sample at the given index.
        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            img (PIL.Image): Image at the given index.
            label (int): Label of the image at the given index.
        """
        img = cv2.imread(self.datalist[idx][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = str(self.datalist[idx][1])

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        label_dict = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
        label = label_dict.get(label, 8)

        return img, label

class Pcam_Dataset(Dataset):
    """
    CRC_Dataset is a custom dataset class for loading and processing CRC images.
    It inherits from the PyTorch Dataset class and overrides the __init__, __len__, and __getitem__ methods.
    """

    def __init__(self, dataset_path, transform=None):
        """
        Initialize the CRC_Dataset with the given dataset_path and optional transform.
        Args:
            dataset_path (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self.datalist = []
        self.label_dict = {'pos': 1, 'neg': 0}
        cls_paths = glob.glob('{}/*/'.format(dataset_path))

        with tqdm(enumerate(sorted(cls_paths)), disable=True) as t:
            for wj, cls_path in t:
                cls_id = str(os.path.split(os.path.dirname(cls_path))[-1])
                patch_pths = glob.glob('{}/*.jpg'.format(cls_path))
                for pth in patch_pths:
                    self.datalist.append((pth, cls_id))
        self.targets = [self.label_dict[img_data[1]] for img_data in self.datalist]

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.datalist)

    def __getitem__(self, idx):
        """
        Get the sample at the given index.
        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            img (PIL.Image): Image at the given index.
            label (int): Label of the image at the given index.
        """
        img = cv2.imread(self.datalist[idx][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = str(self.datalist[idx][1])

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        label_dict = {'pos': 1, 'neg': 0}
        label = label_dict.get(label, 2)

        return img, label
def create_data_loader(dataset, args, sampler=None, mode='test', drop_last=False):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              sampler=sampler,
                                              shuffle=False,  # if mode!='test' else False
                                              num_workers=args.dataloader_num_workers,
                                              pin_memory=True,
                                              drop_last=drop_last)
    return data_loader


def get_dataloader(args, mode):
    if mode == 'linear_eval':
        image_size = 224
        from albumentations.pytorch import ToTensorV2
        import albumentations as A
        transforms_train = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.RandomBrightness(limit=0.2, p=0.75),  # 随机的亮度变化
            A.RandomContrast(limit=0.2, p=0.75),  # 随机的对比度
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),  # 高斯模糊
                A.GaussNoise(var_limit=(5.0, 30.0)),  # 加高斯噪声
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),  # 桶形 / 枕形畸变
                A.GridDistortion(num_steps=5, distort_limit=1.),  # 网格畸变
                A.ElasticTransform(alpha=3),  # 弹性变换
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),  # 对输入图像应用限制对比度自适应直方图均衡化
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # 随机改变图像的色调，饱和度，亮度。
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            # 应用仿射变换：平移、缩放、旋转
            A.Resize(image_size, image_size),
            A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),

            A.Normalize(),
            ToTensorV2(),

        ])

        transforms_val = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ])

        traindir = os.path.join(args.dataset_path, f'NCT-CRC-HE-100K')
        testdir = os.path.join(args.dataset_path, 'CRC-VAL-HE-7K')

        train_dataset = CRC_Dataset(
            traindir,
            transform=transforms_train
        )

        test_dataset = CRC_Dataset(
            testdir,
            transform=transforms_val
        )


        num_train = len(train_dataset.datalist)
        indices = list(range(num_train))

        np.random.shuffle(indices)
        train_idx = indices
        train_idx = np.random.choice(train_idx, int(args.labeled_train * len(train_idx)))

        train_sampler = SubsetRandomSampler(train_idx)

        train_loader = create_data_loader(train_dataset, args, sampler=train_sampler, mode='train')
        test_loader = create_data_loader(test_dataset, args, sampler=None, mode='test')

        return train_loader, test_loader
    elif mode == 'compute_feature':
        image_size = 224
        from albumentations.pytorch import ToTensorV2
        import albumentations as A

        transforms_val = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ])

        train_path = '/home/lijunjian/data/Kather_Multi_Class/NCT-CRC-HE-100K'
        test_path = '/home/lijunjian/data/Kather_Multi_Class/CRC-VAL-HE-7K'

        traindir = train_path
        testdir = test_path

        print('traindir', traindir)
        print('testdir', testdir)
        train_dataset = CRC_Dataset(
            traindir,
            transform=transforms_val
        )

        test_dataset = CRC_Dataset(
            testdir,
            transform=transforms_val
        )

        num_train = len(train_dataset.datalist)
        indices = list(range(num_train))

        train_idx = indices

        train_sampler = SubsetRandomSampler(train_idx)

        train_loader = create_data_loader(train_dataset, args, sampler=train_sampler, mode='train')
        test_loader = create_data_loader(test_dataset, args, sampler=None, mode='test')

        return train_loader, test_loader
    else:
        image_size = 224
        from albumentations.pytorch import ToTensorV2
        import albumentations as A
        transforms_train = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.RandomBrightness(limit=0.2, p=0.75),  # 随机的亮度变化
            A.RandomContrast(limit=0.2, p=0.75),  # 随机的对比度
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),  # 高斯模糊
                A.GaussNoise(var_limit=(5.0, 30.0)),  # 加高斯噪声
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),  # 桶形 / 枕形畸变
                A.GridDistortion(num_steps=5, distort_limit=1.),  # 网格畸变
                A.ElasticTransform(alpha=3),  # 弹性变换
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),  # 对输入图像应用限制对比度自适应直方图均衡化
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # 随机改变图像的色调，饱和度，亮度。
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            # 应用仿射变换：平移、缩放、旋转
            A.Resize(image_size, image_size),
            A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),

            A.Normalize(),
            ToTensorV2(),

        ])

        transforms_val = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ])

        traindir = os.path.join(args.dataset_path, f'NCT-CRC-HE-100K')
        testdir = os.path.join(args.dataset_path, 'CRC-VAL-HE-7K')

        train_dataset = CRC_Dataset(
            traindir,
            transform=transforms_train
        )

        val_dataset = CRC_Dataset(
            traindir,
            transform=transforms_val
        )

        test_dataset = CRC_Dataset(
            testdir,
            transform=transforms_val
        )

        num_train = len(train_dataset.datalist)
        indices = list(range(num_train))
        split = int(np.floor(args.validation_split * num_train))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_idx = np.random.choice(train_idx, int(args.labeled_train * len(train_idx)))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = create_data_loader(train_dataset, args, sampler=train_sampler, mode='train')
        val_loader = create_data_loader(val_dataset, args, sampler=val_sampler, mode='val')
        test_loader = create_data_loader(test_dataset, args, sampler=None, mode='test')

        return train_loader, val_loader, test_loader


def get_dataloader_pcam(args, mode):
    if mode == 'linear_eval':
        image_size = 224
        from albumentations.pytorch import ToTensorV2
        import albumentations as A
        transforms_train = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.RandomBrightness(limit=0.2, p=0.5),  # 随机的亮度变化
            A.RandomContrast(limit=0.2, p=0.5),  # 随机的对比度

            A.CLAHE(clip_limit=2.0, p=0.7),  # 对输入图像应用限制对比度自适应直方图均衡化
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # 随机改变图像的色调，饱和度，亮度。
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
            # 应用仿射变换：平移、缩放、旋转
            A.Resize(image_size, image_size),
            # A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),

            A.Normalize(),
            ToTensorV2(),

        ])

        transforms_val = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ])

        traindir = os.path.join(args.dataset_path, f'train')
        testdir = os.path.join(args.dataset_path, 'test')

        train_dataset = Pcam_Dataset(
            traindir,
            transform=transforms_train
        )

        test_dataset = Pcam_Dataset(
            testdir,
            transform=transforms_val
        )

        num_train = len(train_dataset.datalist)
        indices = list(range(num_train))

        train_idx = indices
        train_idx = np.random.choice(train_idx, int(args.labeled_train * len(train_idx)))

        train_sampler = SubsetRandomSampler(train_idx)

        train_loader = create_data_loader(train_dataset, args, sampler=train_sampler, mode='train')
        test_loader = create_data_loader(test_dataset, args, sampler=None, mode='test')

        return train_loader, test_loader

    elif mode == 'compute_feature':
        image_size = 224
        from albumentations.pytorch import ToTensorV2
        import albumentations as A

        transforms_val = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ])

        train_path = '/home/lijunjian/data/pcam/train'
        test_path = '/home/lijunjian/data/pcam/val'

        traindir = train_path
        testdir = test_path

        print('traindir', traindir)
        print('testdir', testdir)
        train_dataset = Pcam_Dataset(
            traindir,
            transform=transforms_val
        )

        test_dataset = Pcam_Dataset(
            testdir,
            transform=transforms_val
        )

        num_train = len(train_dataset.datalist)
        indices = list(range(num_train))

        train_idx = indices
        train_idx = np.random.choice(train_idx, int(args.labeled_train * len(train_idx)))

        train_sampler = SubsetRandomSampler(train_idx)

        train_loader = create_data_loader(train_dataset, args, sampler=train_sampler, mode='train')
        test_loader = create_data_loader(test_dataset, args, sampler=None, mode='test')

        return train_loader, test_loader
    else:
        image_size = 224
        from albumentations.pytorch import ToTensorV2
        import albumentations as A
        transforms_train = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.RandomBrightness(limit=0.2, p=0.75),  # 随机的亮度变化
            A.RandomContrast(limit=0.2, p=0.75),  # 随机的对比度

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),  # 桶形 / 枕形畸变
                A.GridDistortion(num_steps=5, distort_limit=1.),  # 网格畸变
                A.ElasticTransform(alpha=3),  # 弹性变换
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),  # 对输入图像应用限制对比度自适应直方图均衡化
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # 随机改变图像的色调，饱和度，亮度。
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            # 应用仿射变换：平移、缩放、旋转
            A.Resize(image_size, image_size),
            A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),

            A.Normalize(),
            ToTensorV2(),

        ])

        transforms_val = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ])

        traindir = os.path.join(args.dataset_path, f'train')
        valdir   = os.path.join(args.dataset_path, f'val')
        testdir = os.path.join(args.dataset_path, 'test')

        train_dataset = Pcam_Dataset(
            traindir,
            transform=transforms_train
        )

        print('train_dataset', len(train_dataset.datalist))

        val_dataset = Pcam_Dataset(
            valdir,
            transform=transforms_val
        )

        test_dataset = Pcam_Dataset(
            testdir,
            transform=transforms_val
        )

        num_train = len(train_dataset.datalist)
        indices = list(range(num_train))
        train_idx = indices
        train_idx = np.random.choice(train_idx, int(args.labeled_train * len(train_idx)))

        train_sampler = SubsetRandomSampler(train_idx)

        train_loader = create_data_loader(train_dataset, args, sampler=train_sampler, mode='train')
        val_loader   = create_data_loader(val_dataset, args, sampler=None, mode='val')
        test_loader  = create_data_loader(test_dataset, args, sampler=None, mode='test')

        return train_loader, val_loader, test_loader
