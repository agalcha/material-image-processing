import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PhaseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=512, train=True, augment=False):
        all_images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        all_masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        if train:
            # exclude test.png from training
            self.image_paths = [
                p for p in all_images
                if "test" not in os.path.basename(p).lower()
            ]
            self.mask_paths = [
                p for p in all_masks
                if "test" not in os.path.basename(p).lower()
            ]
        else:
            self.image_paths = all_images
            self.mask_paths = all_masks

        assert len(self.image_paths) == len(self.mask_paths), "Mismatch images/masks"
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # horizontal flip
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

            # vertical flip
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

            # 0, 90, 180, 270 degree rotation
            if np.random.rand() < 0.5:
                k = np.random.randint(4)
                img = np.rot90(img, k, (0, 1)).copy()
                mask = np.rot90(mask, k, (0, 1)).copy()

            # very mild brightness jitter
            if np.random.rand() < 0.5:
                factor = 0.9 + 0.2 * np.random.rand()  # 0.9–1.1
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        img_t = torch.from_numpy(img).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img_t, mask_t
