import torch
from torch.utils.data import Dataset
import numpy as np

class ABSADataset(Dataset):

    def __init__(self, path, input_list):
        super(ABSADataset, self).__init__()
        self.data = {}
        if isinstance(path, str):
            data = np.load(path)
            for key, value in data.items():
                self.data[key] = torch.tensor(value).long()
        else:
            for p in path:
                data = np.load(p)
                for key, value in data.items():
                    if key not in self.data:
                        self.data[key] = []
                    self.data[key].append(torch.tensor(value).long())
            for key, value in self.data.items():
                self.data[key] = torch.cat(self.data[key], 0)
        self.len = self.data['label'].size(0)
        self.input_list = input_list

    def __getitem__(self, index):
        return_value = []
        for input in self.input_list:
            return_value.append(self.data[input][index])
        return_value.append(self.data['label'][index])
        return return_value

    def __len__(self):
        return self.len
