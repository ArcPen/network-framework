from torch.utils.data import Dataset
from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

##### Data #####

class MNISTDataset(Dataset):
    def __init__(self, root, mode='train'):
        root = Path(root)
        self.load_data(root, mode)
        self.root = root.as_posix()

    def load_data(self, root, mode):
        if mode == "train":
            file = "train.csv"
        elif mode == "test":
            file = "test.csv"
        else:
            raise ValueError(f"Mode {mode} not recognized")
        csv_file = root / file
        label = []
        data_arr = []
        with open(csv_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if mode == "train":
                    label.append(int(row.pop(0)))
                elif mode == "test":
                    label.append(0)
                data_arr.append([int(i) for i in row])
                
        data_arr = np.array(data_arr)

        num_classes = len(set(label))
        eye = np.eye(num_classes)
        label = np.array([eye[i] for i in label]) # one-hot encoding
        
        self.label = label
        self.data = data_arr

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.label[idx]).float()
        data_dict = OrderedDict(
            img=img,
            label=label,
            metadata=OrderedDict(
                idx=idx+1,
            )
        )
        return data_dict

    def __len__(self):
        return len(self.label)

    def __repr__(self):
        return f'MNISTDataset(root={self.root})'
    
