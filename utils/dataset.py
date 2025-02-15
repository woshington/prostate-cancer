from typing import Any

from torch.utils.data import Dataset
from skimage import io as skio
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform, BaseTransformInitSchema

class PandasDataset(Dataset):
    def __init__(self, image_dir, dataframe, transforms=None):
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.image_id.strip()

        file_path = f"{self.image_dir}/{img_id}.jpg"
        image = skio.imread(file_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), img_id


class RemovePenMarkAlbumentations(ImageOnlyTransform):
    def __init__(self):
        super().__init__(p=1)

    @staticmethod
    def calculate_channel_sums(image):
        red_sum = np.sum(image[:, :, 0])
        green_sum = np.sum(image[:, :, 1])
        blue_sum = np.sum(image[:, :, 2])
        return red_sum, green_sum, blue_sum

    def analyze_histogram(self, image, threshold):
        red_sum, green_sum, blue_sum = self.calculate_channel_sums(image)

        green_dominance = green_sum > threshold * red_sum
        blue_dominance = blue_sum > threshold * red_sum

        return green_dominance or blue_dominance

    def apply(self, img, **params: Any):
        chip_size = (16, 16)
        overlap = 0

        height, width = img.shape[:2]
        chip_h, chip_w = chip_size

        for y in range(0, height, chip_h - overlap):
            for x in range(0, width, chip_w - overlap):
                chip = img[y:y + chip_h, x:x + chip_w]

                if chip.shape[0] < chip_h or chip.shape[1] < chip_w:
                    padded_chip = np.zeros((chip_h, chip_w, img.shape[2]), dtype=img.dtype)
                    padded_chip[:chip.shape[0], :chip.shape[1]] = chip
                    chip = padded_chip

                if self.analyze_histogram(chip, threshold=1):
                    img[y:y + chip_h, x:x + chip_w] = 255

        return img