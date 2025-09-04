import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from config import DefaultConfig
opt = DefaultConfig()


class ItemDataset(Dataset):
    def __init__(self, data_size, seq_len):
        self.data_size = data_size
        self.seq_len = seq_len
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()
        return sample

    def _generate_data(self):
        """
        :return: Set of items_list
        """
        items_list = []
        data_iter = tqdm(range(self.data_size), unit='data')

        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data boxes %i/%i' % (i + 1, self.data_size))
            a = np.c_[np.random.randint(low=opt.L / 10, high=opt.L / 2 + 1, size=(self.seq_len, 1)), np.random.randint(
                low=opt.W / 10, high=opt.W / 2 + 1,
                size=(
                    self.seq_len, 1))]
            items_list.append(np.c_[a, np.random.randint(low=min(opt.L / 10, opt.W / 10),
                                                         high=max(opt.L / 2 + 1, opt.W / 2 + 1),
                                                         size=(self.seq_len, 1))])
        return items_list





