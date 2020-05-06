# -*- coding: utf-8 -*-

import math

from torch.utils.data import Dataset


def split_list(list_to_split, target_n, result):
    if (len(list_to_split) / target_n) % 1 == 0:
        n = int(len(list_to_split) / target_n)
        # print(n)
        i = 0
        while i < len(list_to_split):
            result.append(list_to_split[i:i + n])
            i += n
            # print(i)
    else:
        n = int(math.floor(len(list_to_split) / target_n))
        result.append(list_to_split[0:n])
        del list_to_split[0:n]
        split_list(list_to_split, target_n - 1, result)


class ThisRankDataset(Dataset):
    """产生数据倾斜，每一个learner中含有固定的label的数据集 - Data skew, each learner owns a certain subset according with fixed labels"""

    def __init__(self, all_data, labels, transform=None):

        img_list = []
        for idx, (img, label) in enumerate(all_data):
            if label in labels:
                img_list.append((img, int(label)))

        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.img_list[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_list)
