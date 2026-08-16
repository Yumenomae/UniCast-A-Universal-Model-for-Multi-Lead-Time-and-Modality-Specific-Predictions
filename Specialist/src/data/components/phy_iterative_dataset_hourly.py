import os
import h5py
import torch
import random
import numpy as np

from glob import glob
from torch.utils.data import Dataset
from typing import Any, Dict, Optional, Tuple, Sequence

def get_data_given_path(
        path: str, variables: Sequence[str]
) -> torch.tensor:
    """Prepare the data given the data path and variables.
    
    """
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables + ['time', 'exist_cmpa']
        } for main_key, group in f.items() if main_key in ['input']}

    x = [data['input'][v] for v in variables]
    return np.stack(x, axis=0), data['input']['exist_cmpa']

def get_data_until_exist_cmpa(
        data_path: str, variables: Sequence[str]
) -> torch.tensor:
    """Prepare the data given the data path and variables.
    
    """
    flag = False
    while not flag:
        path = random.choice(data_path)
        inp_data, exist_cmpa = get_data_given_path(path, variables)
        flag = exist_cmpa
    return path, inp_data

def get_target_path(
        data_dir: str, year: str, inp_file_idx: int, steps: int
) -> str:
    """get the target path according to the input data index and forward steps.
    
    :param year: current year
    :param inp_file_idx: file index of the input in the current year
    :param steps: number of steps forward
    """
    target_file_idx = inp_file_idx + steps
    target_path = os.path.join(data_dir, f'{year}_{target_file_idx:04}.h5')
    if not os.path.exists(target_path):
        for i in range(steps):
            target_file_idx = inp_file_idx + i
            target_path = os.path.join(data_dir, f'{year}_{target_file_idx:04}.h5')
            if os.path.exists(target_path):
                max_step_forward = i
        remaining_steps = steps - max_step_forward
        next_year = year + 1
        target_path = os.path.join(data_dir, f'{next_year}_{remaining_steps-1:04}.h5')
    return target_path

class MultiStepRandomizedDataset(Dataset):
    """Training dataset.

    The training dataset consists of 1 input and multiple desired outputs at multiple intervals (randomly chosen).
    
    """
    def __init__(
        self,
        data_dir,
        variables,
        inp_transform,
        train_lead_times=[1, 3, 6, 12],
        # num_lead_times=2,
    ):
        super().__init__()
    

        self.data_dir = data_dir
        self.variables = variables
        self.inp_transform = inp_transform
        self.lead_times_set = train_lead_times
        self.assist_times_set = [lead_time + 1 for lead_time in train_lead_times]
        # self.num_lead_times = num_lead_times
        
        file_paths = glob(os.path.join(data_dir, '*.h5'))
        file_paths = sorted(file_paths)
        self.inp_file_paths = file_paths[:-max(self.assist_times_set)] # the last few points do not have ground-truth
        self.file_paths = file_paths

        self.inp_file_paths = self.inp_file_paths
        print('The Training Dataset is [2018, 2019, 2020, 2021]')
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data, exist_cmpa = get_data_given_path(inp_path, self.variables)
        if not exist_cmpa:
            inp_path, inp_data = get_data_until_exist_cmpa(self.inp_file_paths, self.variables)

        # randomly choose an interval and get the corresponding ground-truths
        # chosen_lead_times = np.random.choice(self.lead_times_set, size=self.num_lead_times, replace=False)
        chosen_lead_times = self.lead_times_set
        assist_lead_times = self.assist_times_set

        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        targets, target_exist_cmpa_set = [], []
        assists, assists_exist_cmpa_set = [], []
        
        # get ground-truths at multiple steps
        for lead_time in chosen_lead_times:
            target_path = get_target_path(self.data_dir, year, inp_file_idx, steps=lead_time)
            target, exist_cmpa = get_data_given_path(target_path, self.variables)
            target = torch.from_numpy(target)
            targets.append(self.inp_transform(target))
            target_exist_cmpa_set.append(torch.from_numpy(exist_cmpa))

        for lead_time in assist_lead_times:
            assist_path = get_target_path(self.data_dir, year, inp_file_idx, steps=lead_time)
            assist, exist_cmpa = get_data_given_path(assist_path, self.variables)
            assist = torch.from_numpy(assist)
            assists.append(self.inp_transform(assist))
            assists_exist_cmpa_set.append(torch.from_numpy(exist_cmpa))
        
        inp_data = torch.from_numpy(inp_data)
        targets = torch.stack(targets, dim=0)
        chosen_lead_times = np.array(chosen_lead_times)
        chosen_lead_times = torch.from_numpy(chosen_lead_times).to(dtype=inp_data.dtype)
        target_exist_cmpa_set = torch.stack(target_exist_cmpa_set)

        assists = torch.stack(assists, dim=0)
        assists_exist_cmpa_set = torch.stack(assists_exist_cmpa_set)
        
        return (
            self.inp_transform(inp_data), # （V, H, W）
            targets, # (T, V, H, W)
            torch.tensor(inp_file_idx),
            chosen_lead_times,
            target_exist_cmpa_set,
            self.variables,
            assists,
            assists_exist_cmpa_set,
        )

class MultiLeadtimeDataset(Dataset):
    """Validation/Test dataset.

    The validation and test datasets consist of 1 input and multiple desired outputs at multiple lead times.
    
    """
    def __init__(
        self,
        data_dir,
        variables,
        transform,
        lead_times_set = [1, 2, 4, 8, 16],
    ):
        super().__init__()

        self.data_dir = data_dir
        self.variables = variables
        self.transform = transform
        self.lead_times_set = lead_times_set
        self.assist_times_set = [lead_time + 1 for lead_time in lead_times_set]
        
        file_paths = glob(os.path.join(data_dir, '*.h5'))
        file_paths = sorted(file_paths)
        max_lead_time = max(*self.assist_times_set) if len(self.assist_times_set) > 1 else self.assist_times_set[0]
        self.inp_file_paths = file_paths[:-max_lead_time] # the last few points do not have ground-truth
        self.file_paths = file_paths

        ############ pilot
        # self.inp_file_paths = [self.inp_file_paths[i] for i in range(0, len(self.inp_file_paths), 10)]
        # self.inp_file_paths = self.inp_file_paths[0:8]

    def __len__(self):
        return len(self.inp_file_paths)
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data, exist_cmpa = get_data_given_path(inp_path, self.variables)

############# Pilot 
        # if not exist_cmpa:
        #     inp_path = self.inp_file_paths[3768]
        #     inp_data, exist_cmpa = get_data_given_path(inp_path, self.variables)

        if not exist_cmpa:
            inp_path = self.inp_file_paths[0]
            inp_data, exist_cmpa = get_data_given_path(inp_path, self.variables)
############# Pilot 
#             
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        dict_target = {}
        dict_exist_cmpa = {}

        dict_assists = {}
        dict_assists_exist_cmpa = {}
        
        # get ground-truth paths at multiple lead times
        for lead_time in self.lead_times_set:
            target_path = get_target_path(self.data_dir, year, inp_file_idx, steps=lead_time)
            target_data, exist_cmpa = get_data_given_path(target_path, self.variables)
            dict_target[lead_time] = target_data
            dict_exist_cmpa[lead_time] = torch.tensor(exist_cmpa)

        inp_data = torch.from_numpy(inp_data)
        dict_target = {lead_time: torch.from_numpy(target) for lead_time, target in dict_target.items()}

        for lead_time in self.assist_times_set:
            assist_path = get_target_path(self.data_dir, year, inp_file_idx, steps=lead_time)
            assist_data, exist_cmpa = get_data_given_path(assist_path, self.variables)
            dict_assists[lead_time] = assist_data
            dict_assists_exist_cmpa[lead_time] = torch.tensor(exist_cmpa)

        dict_assists = {lead_time: torch.from_numpy(assist) for lead_time, assist in dict_assists.items()}    

        
        return (
            self.transform(inp_data), # （V, H, W）
            torch.tensor(inp_file_idx),
            {lead_time: self.transform(target) for lead_time, target in dict_target.items()},
            dict_exist_cmpa,
            self.variables,
            {lead_time: self.transform(assist) for lead_time, assist in dict_assists.items()},
            dict_assists_exist_cmpa,
        )