import os
import h5py
import torch
import numpy as np

from glob import glob
from torch.utils.data import Dataset
from typing import Any, Dict, Optional, Tuple, Sequence

class EmbeddingDataset(Dataset):
    """Embedding dataset.

    The embedding dataset consists of embedding data for different time (hour of year).
    
    """
    def __init__(
        self,
        data_dir,
    ):
        super().__init__()
        self.data_dir = data_dir
        
        file_paths = glob(os.path.join(data_dir, '*.npy'))
        file_paths = sorted(file_paths)
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        embedding_path = self.file_paths[index]
        embedding = np.load(embedding_path) # (d, 441, 845)
        
        return (
            torch.from_numpy(embedding), # （V, H, W）
        )