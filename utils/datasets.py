import glob
import os

import cv2
from torch.utils.data import Dataset
from tqdm import tqdm


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
        cls_paths = glob.glob('{}/*/'.format(dataset_path))
        with tqdm(enumerate(sorted(cls_paths)), disable=True) as t:
            for wj, cls_path in t:
                cls_id = str(os.path.split(os.path.dirname(cls_path))[-1])
                patch_pths = glob.glob('{}/*.tif'.format(cls_path))
                for pth in patch_pths:
                    self.datalist.append((pth, cls_id))

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

        if self.transform is not None:
            img = self.transform(img)

        return img, 1
