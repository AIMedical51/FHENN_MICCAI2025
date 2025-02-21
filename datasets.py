import os
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from PIL import Image
class HAM10000Dataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]['image_id'] + '.jpg'
        label = self.annotations.iloc[idx]['label']

        image = None
        for directory in self.img_dir:
            img_path = os.path.join(directory, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                break

        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found in specified directories.")

        if self.transform:
            image = self.transform(image)
        return image, label



# Define custom dataset for OrganSMNIST
class OrganSMNISTDataset(Dataset):
    def __init__(self, images, coarse_labels,fine_labels, transform=None):
        self.images = images
        self.coarse_labels = coarse_labels
        self.fine_labels = fine_labels
        self.transform = transform

    def __len__(self):
        return len(self.coarse_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        coarse_label = self.coarse_labels[idx]
        fine_label = self.fine_labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        coarse_label = torch.tensor(coarse_label, dtype=torch.long)
        fine_label = torch.tensor(fine_label, dtype=torch.long)
        return image, coarse_label, fine_label
class BASEOrganSMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long).squeeze()
        return image, label



class OrganSMNISTDataset(Dataset):
    def __init__(self, images, coarse_labels,fine_labels, transform=None):
        self.images = images
        self.coarse_labels = coarse_labels
        self.fine_labels = fine_labels
        self.transform = transform

    def __len__(self):
        return len(self.coarse_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        coarse_label = self.coarse_labels[idx]
        fine_label = self.fine_labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        coarse_label = torch.tensor(coarse_label, dtype=torch.long)
        fine_label = torch.tensor(fine_label, dtype=torch.long)
        return image, coarse_label, fine_label


# Define custom dataset for OrganSMNIST
class SingleOrganSMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long).squeeze()
        return image, label

