import os
import random

import jpegio as jio
import numpy as np
import scipy
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms


def label_smoothing(labels, factor=0.05):
    # 对于二分类任务，标签将是0或1
    # 使用Label Smoothing技术调整标签(针对这个任务,在0.01到0.05之间选取)
    smooth_labels = (1 - factor) * labels + factor * (1 - labels)
    return smooth_labels


# Define the image and label transforms at the top level of the module
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Define transform_fn at the top level of the module
def transform_fn(image, label):
    image = transforms.ToTensor()(image).unsqueeze(0)
    label = torch.from_numpy(np.array(label)).float()
    label = label_smoothing(label)
    return image, label


# Now get_transforms simply returns transform_fn
def get_transforms():
    return transform_fn


# 定义可同时应用于图像和标签的随机变换
def random_transforms(image, label, augmentation_prob):
    # 随机旋转
    RotationDegree = random.choice([0, 90, 180, 270])
    if random.random() <= augmentation_prob:
        image = transforms.functional.rotate(image, RotationDegree)
        label = transforms.functional.rotate(label, RotationDegree)

    # 随机水平翻转
    if random.random() <= augmentation_prob:
        image = transforms.functional.hflip(image)
        label = transforms.functional.hflip(label)

    # 随机垂直翻转
    if random.random() <= augmentation_prob:
        image = transforms.functional.vflip(image)
        label = transforms.functional.vflip(label)

    return image, label


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None, augmentation_prob=0.5, dct_domain=False,
                 embedding_rate=None, sample_ratio=1.0):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.augmentation_prob = augmentation_prob
        self.dct_domain = dct_domain
        self.embedding_rate = embedding_rate
        self.sample_ratio = sample_ratio

        if mode == 'train' or mode == 'val':
            self.images_dir = os.path.join(data_dir, mode, 'images')
            self.labels_dir = os.path.join(data_dir, mode, 'labels')
            self.filenames = [f.split('.')[0] for f in os.listdir(self.images_dir)]
            self.sample_indices = random.sample(range(len(self.filenames)),
                                                int(len(self.filenames) * self.sample_ratio))

        elif mode == 'test' and embedding_rate is not None:
            self.images_dir = os.path.join(data_dir, mode, f'images_{embedding_rate}')
            self.labels_dir = os.path.join(data_dir, mode, f'labels_{embedding_rate}')
            self.filenames = [f.split('.')[0] for f in os.listdir(self.images_dir)]
            self.sample_indices = random.sample(range(len(self.filenames)), int(len(self.filenames) * sample_ratio))

    def dct_cfsi(self, dct):
        # Co-frequency sub-images
        batch_size, _, col, row = dct.shape
        subdct = torch.zeros((batch_size, batch_size * 64, col // 8, row // 8), dtype=dct.dtype, device=dct.device)

        for i in range(batch_size):
            for j in range(64):
                subdct[i, j + i * 64, :, :] = dct[i, :, j // 8:col:8, j % 8:row:8]

        return subdct

    def __len__(self):
        return int(len(self.filenames) * self.sample_ratio)

    def __getitem__(self, idx):
        # sample_idx = self.sample_indices[idx]
        if self.mode == 'train' or self.mode == 'val':
            img_name = os.path.join(self.images_dir, self.filenames[idx] + '.jpg')
            label_name = os.path.join(self.labels_dir, self.filenames[idx])
        elif self.mode == 'test':
            img_name = os.path.join(self.images_dir, self.filenames[idx] + '.jpg')
            label_name = os.path.join(self.labels_dir, self.filenames[idx])

        if self.dct_domain:
            # Read image in DCT domain
            img = jio.read(img_name)
            dct = img.coef_arrays[0].astype(np.float32)
            image = Image.fromarray(dct)  # Convert to PyTorch tensor
        else:
            # Read image in spatial domain
            image = Image.open(img_name)

        # 读取.mat文件
        label = scipy.io.loadmat(label_name)['label']
        # 将label从NumPy数组转换为PIL图像，方便数据增强
        label = label.reshape((512, 512), order='F')
        label = Image.fromarray(label)

        if self.mode == 'train':
            image, label = random_transforms(image, label, self.augmentation_prob)  # 应用随机变换

        # 对图像和标签进行数据增强
        image, label = self.transform(image, label)

        if self.dct_domain:
            image = self.dct_cfsi(image)

        return torch.squeeze(image), label


def get_loader(image_path=None, batch_size=1, num_workers=2, mode='train', augmentation_prob=0.6, dct_domain=False,
               sample_ratio=1.0, embedding_rate=None):
    """Builds and returns Dataloader."""

    dataset = CustomDataset(data_dir=image_path, mode=mode, transform=get_transforms(),
                            augmentation_prob=augmentation_prob, dct_domain=dct_domain,
                            sample_ratio=sample_ratio, embedding_rate=embedding_rate)
    sampler = SubsetRandomSampler(np.random.choice(len(dataset), size=len(dataset), replace=False))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  sampler=sampler)
    return data_loader


if __name__ == '__main__':
    data_path = '../StegoDataSets/JstegM'
    train_loader = get_loader(image_path=data_path, batch_size=32, num_workers=4, mode='train',
                              augmentation_prob=0.6, dct_domain=False)
    val_loader = get_loader(image_path=data_path, batch_size=32, num_workers=4, mode='val',
                            augmentation_prob=0.6, dct_domain=False)
    test_loader_005 = get_loader(image_path=data_path, batch_size=1, num_workers=4, mode='test',
                                 augmentation_prob=0.6, dct_domain=True, embedding_rate='0.05')
    test_loader_09 = get_loader(image_path=data_path, batch_size=32, num_workers=4, mode='test',
                                augmentation_prob=0.6, dct_domain=False, embedding_rate='0.9')
