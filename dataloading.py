"""
Pytorch data loader for the converted TFRecord data
"""
import torch
from torch.utils import data
from torch import Tensor
import glob
import numpy as np

class DatasetNew(torch.utils.data.IterableDataset):
    def __init__(self, data_loc='/home/ubuntu/data/gridcells/', batch_size=100, n_sample=1000000, seed=None, shuffle=True):
        super(Dataset).__init__()
        self.file_list = glob.glob(data_loc+'*')
        self.seed = seed
        n_sample = min(int(n_sample), 1000000)
        batch_size = min(int(batch_size), n_sample)
        n_batch = n_sample//batch_size
        n_batch_per_file = 10000//batch_size
        n_batch = min(n_batch, n_batch_per_file*100)
        n_batch_per_file = min(n_batch, n_batch_per_file)
        self.n_sample = n_batch*batch_size
        self.batch_size = batch_size
        self.n_batch_per_file = n_batch_per_file
        self.n_batch = n_batch
        self.n_sample_per_file = batch_size*n_batch_per_file
        self.n_file = int(np.ceil(n_batch/n_batch_per_file))
        self.rng = np.random.default_rng(seed)
        self.file_ids = self.rng.choice(len(self.file_list), size=(self.n_file,), replace=False)
        self.indices = {file_id:self.rng.choice(10000, size=(self.n_sample_per_file,), replace=False)
                        for file_id in self.file_ids}
        self.shuffle = shuffle

    def __len__(self):
        return self.n_sample

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        file_ids = self.file_ids
        if self.shuffle:
            self.rng.shuffle(file_ids)
        if worker_info is not None:
            worker_id = worker_info.id
            per_worker = int(np.ceil(self.n_file/float(worker_info.num_workers)))
            iter_start = worker_id*per_worker
            iter_end = min(iter_start+per_worker, self.n_file)
            file_ids = file_ids[iter_start:iter_end]
            # print(f'Worker {worker_id} loading files {file_ids}')
        for file_id in file_ids:
            data = torch.load(self.file_list[file_id])
            indices = self.indices[file_id]
            if self.shuffle:
                self.rng.shuffle(indices)
            indices = indices.reshape((-1, self.batch_size))
            for index in indices:
                X = (torch.Tensor(data['init_pos'][index]), torch.Tensor(data['init_hd'][index]), torch.Tensor(data['ego_vel'][index]))
                y = (torch.Tensor(data['target_pos'][index]), torch.Tensor(data['target_hd'][index]))
                yield X, y

# class Dataset(torch.utils.data.IterableDataset):
#     def __init__(self, data_loc='/home/ubuntu/data/gridcells/', batch_size=100, n_sample=1000000, seed=None):
#         super(Dataset).__init__()
#         self.file_list = glob.glob(data_loc+'*')
#         self.data = None
#         self.seed = seed
#         n_sample = min(int(n_sample), 1000000)
#         batch_size = min(int(batch_size), n_sample)
#         n_batch = n_sample//batch_size
#         n_batch_per_file = 10000//batch_size
#         n_batch = min(n_batch, n_batch_per_file*100)
#         n_batch_per_file = min(n_batch, n_batch_per_file)
#         self.n_sample = n_batch*batch_size
#         self.batch_size = batch_size
#         self.n_batch_per_file = n_batch_per_file
#         self.n_batch = n_batch
#         self.n_sample_per_file = batch_size*n_batch_per_file
#         self.n_file = int(np.ceil(n_batch/n_batch_per_file))

#     def __len__(self):
#         return self.n_sample

#     def __iter__(self):
#         if self.seed:
#             torch.manual_seed(self.seed)
#         file_ids = torch.randint(high=len(self.file_list), size=(self.n_file, ))
#         file_index = 0
#         for batch_index in range(self.n_batch):
#             batch_index_in_file = batch_index%self.n_batch_per_file
#             if batch_index_in_file==0:
#                 self.data = torch.load(self.file_list[file_ids[file_index]])
#                 indices = torch.randint(high=10000, size=(self.n_sample_per_file, )).reshape((-1, self.batch_size))
#             index = indices[batch_index_in_file]
#             X = (torch.Tensor(self.data['init_pos'][index]), torch.Tensor(self.data['init_hd'][index]), torch.Tensor(self.data['ego_vel'][index]))
#             y = (torch.Tensor(self.data['target_pos'][index]), torch.Tensor(self.data['target_hd'][index]))
#             yield X, y

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_loc='/home/ubuntu/data/gridcells/', batch_size=256):
        'Initialization'
        self.file_list = glob.glob(data_loc + '*' )
        self.j = 0
        self.temp = 0
        self.data = None
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the total number of samples'
        return int(1e6)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        index = index % 10000

        loaded_data = self.loaded_data()
        X = (
            Tensor(loaded_data['init_pos'][index]) ,
            Tensor(loaded_data['init_hd'][index]),
            Tensor(loaded_data['ego_vel'][index])
            )

        y = (Tensor(loaded_data['target_pos'][index]) ,
                Tensor(loaded_data['target_hd'][index]))
        return X, y

    def loaded_data(self):
        if self.data == None or self.j % self.batch_size == 0:
            file_id = torch.randint(high=len(self.file_list), size=(1, ))
            ID = self.file_list[file_id]
            self.data = torch.load(ID)

        self.j += 1
        return self.data
