import os
import time
from typing import List, Callable, Any
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TripletDataset(Dataset):
    def __init__(self,
                 root: str,
                 condition_indices: List[int]=None,
                 split: str="train",
                 shuffle: bool=True,
                 n_triplets: int=None,
                 transform: Callable=None,
                 loader: Callable[[str], Any]=default_image_loader):
        """
        Args:
            root (string): Root directory path.
            condition_indices (list[int]): A list of indices of allowed conditions.
            split (string): "train" | "val" | "test"
            shuffle (bool): If true, shuffle the triplets.
            n_triplets (int): Number of triplets to use. `None` to use all triplets.
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            loader (callable): A function to load a sample given its path.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.loader = loader

        # read conditions
        self.conditions = np.loadtxt(os.path.join(root, "conditions.txt"), dtype=str)
        if condition_indices:
            self.conditions = self.conditions[condition_indices]
            self.condition_indices = condition_indices
        else:
            self.condition_indices = list(range(len(self.conditions)))

        # read product ids
        self.product_ids = np.loadtxt(os.path.join(root, "product_ids.txt"), dtype=int)
        self.idx_to_product_id = {i: str(k) for i, k in enumerate(self.product_ids)}

        # read triplets
        _triplets = []
        for condition, idx in zip(self.conditions, self.condition_indices):
            t = np.loadtxt(os.path.join(self.root, "triplets", f"{condition}_{split}.txt"), dtype=int)
            t = np.hstack([t, np.full((len(t), 1), idx)])
            _triplets.append(t)
        self.triplets = np.vstack(_triplets)
        if shuffle:
            np.random.shuffle(self.triplets)
        if n_triplets:
            self.triplets = self.triplets[:n_triplets]

        print(f"TripletDataset split={self.split} conditions={self.conditions} triplets={len(self.triplets)}")

    def load_image(self, idx):
        product_id: str = self.idx_to_product_id[idx]
        path = os.path.join(self.root, "images", product_id[:2], f"{product_id}.jpg")
        img = self.loader(path)
        return img

    def __getitem__(self, i):
        idx1, idx2, idx3, c = self.triplets[i]
        img1 = self.load_image(idx1)
        img2 = self.load_image(idx2)
        img3 = self.load_image(idx3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img1, img2, img3, c

    def __len__(self):
        return len(self.triplets)

dataset = TripletDataset("./data/farfetch", split="train")
print(f"len={len(dataset)}")
a = dataset[0]
s = time.perf_counter()
b = dataset[1]
e = time.perf_counter()
print(f"time={e-s:.4f}s")