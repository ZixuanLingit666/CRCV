from __future__ import absolute_import
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if isinstance(self.transform, list):
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)       # todo 灰度图
            img = torch.cat([img1, img2], dim=0)
        else:
            img = self.transform(img)

        return img, fname, pid, camid, index
