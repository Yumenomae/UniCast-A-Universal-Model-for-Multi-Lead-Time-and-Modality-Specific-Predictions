import os
from typing import Any, Dict, Optional, Tuple, List

import torch
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from src.data.components.phy_iterative_dataset_hourly import MultiStepRandomizedDataset, MultiLeadtimeDataset

def collate_fn_train(
    batch: Tuple[torch.tensor, torch.tensor, torch.tensor, int, bool, List[str]],
) -> Tuple[torch.tensor, torch.tensor, List[str]]:
    """Prepare the input and output data for training`.

    :param batch: A single batch of data.
    """
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, V, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))]) # B, T, V, H, W
    # out_transform_mean = torch.stack([batch[i][2] for i in range(len(batch))]) # B, V
    # out_transform_std = torch.stack([batch[i][3] for i in range(len(batch))]) # B, V
    hour_of_year = torch.stack([batch[i][2] for i in range(len(batch))]) # B,
    lead_times = torch.stack([batch[i][3] for i in range(len(batch))]) # B, T
    exist_cmpa = torch.stack([batch[i][4] for i in range(len(batch))]) # B, T
    variables = batch[0][5]
    assists = torch.stack([batch[i][6] for i in range(len(batch))]) # B, T, V, H, W
    assists_exist_cmpa = torch.stack([batch[i][7] for i in range(len(batch))]) # B, T

    return inp, out, hour_of_year, lead_times, exist_cmpa, variables, assists, assists_exist_cmpa

def collate_fn_val(
    batch: Tuple[torch.tensor, int, dict, List[str]],
) -> Tuple[torch.tensor, torch.tensor, List[str]]:
    """Prepare the input and output data for validation/test`.

    :param batch: A single batch of data.
    """
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, V, H, W
    hour_of_year = torch.tensor([batch[i][1] for i in range(len(batch))]) # B,

    out_dicts = [batch[i][2] for i in range(len(batch))]
    exist_cmpa_dicts = [batch[i][3] for i in range(len(batch))] 
    val_lead_times = out_dicts[0].keys()
    out, exist_cmpa = {}, {}
    for lead_time in val_lead_times:
        out[lead_time] = torch.stack([out_dicts[i][lead_time] for i in range(len(batch))])
        exist_cmpa[lead_time] = torch.tensor([exist_cmpa_dicts[i][lead_time] for i in range(len(batch))])

    variables = batch[0][4]

    assists_dicts = [batch[i][5] for i in range(len(batch))]
    assists_exist_cmpa_dicts = [batch[i][6] for i in range(len(batch))] 
    assist_lead_times = assists_dicts[0].keys()
    assists, assists_exist_cmpa = {}, {}
    for lead_time in assist_lead_times:
        assists[lead_time] = torch.stack([assists_dicts[i][lead_time] for i in range(len(batch))])
        assists_exist_cmpa[lead_time] = torch.tensor([assists_exist_cmpa_dicts[i][lead_time] for i in range(len(batch))])


    
    return inp, out, hour_of_year, exist_cmpa, variables, assists, assists_exist_cmpa

class DataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        variables: List[str],
        train_lead_times: List[int],
        val_lead_times: List[int],
        # num_lead_times: int, 
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        normalize_mean = dict(np.load(os.path.join(data_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(data_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
        assist_normalize_mean = dict(np.load(os.path.join(data_dir, "normalize_assist_mean.npz")))
        assist_normalize_mean = assist_normalize_mean['airColumn_h']

        assist_normalize_std = dict(np.load(os.path.join(data_dir, "normalize_assist_std.npz")))
        assist_normalize_std = assist_normalize_std['airColumn_h']

        self.transforms = transforms.Normalize(normalize_mean, normalize_std)
        self.assist_transforms = transforms.Normalize(assist_normalize_mean, assist_normalize_std)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def get_lat_lon(self) -> Tuple[np.array, np.array]:
        lat = np.load(os.path.join(self.hparams.data_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.data_dir, "lon.npy"))
        return lat, lon
    
    def get_transforms(self):
        return self.transforms

    def get_assist_transforms(self):
        return self.assist_transforms
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
                self.data_train = MultiStepRandomizedDataset(
                    data_dir=os.path.join(self.hparams.data_dir, 'train'),
                    variables=self.hparams.variables,
                    inp_transform=self.transforms,
                    # steps=self.hparams.steps,
                    train_lead_times=self.hparams.train_lead_times,
                )

                self.data_val = MultiLeadtimeDataset(
                    data_dir=os.path.join(self.hparams.data_dir, 'val'),
                    variables=self.hparams.variables,
                    transform=self.transforms,
                    lead_times_set=self.hparams.val_lead_times,
                    # data_freq=self.hparams.data_freq
                )

                self.data_test = MultiLeadtimeDataset(
                    data_dir=os.path.join(self.hparams.data_dir, 'test'),
                    variables=self.hparams.variables,
                    transform=self.transforms,
                    lead_times_set=self.hparams.val_lead_times,
                    # data_freq=self.hparams.data_freq
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn_train,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn_val,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn_val,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    # def state_dict(self) -> Dict[Any, Any]:
    #     """Called when saving a checkpoint. Implement to generate and save the datamodule state.

    #     :return: A dictionary containing the datamodule state that you want to save.
    #     """
    #     return {}

    # def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    #     """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
    #     `state_dict()`.

    #     :param state_dict: The datamodule state returned by `self.state_dict()`.
    #     """
    #     pass


if __name__ == "__main__":
    _ = DataModule()
