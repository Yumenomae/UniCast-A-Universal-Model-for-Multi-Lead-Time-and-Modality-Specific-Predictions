from typing import Any, Dict, Tuple, Dict, List
import torch
import numpy as np
from lightning import LightningModule
from torchvision.transforms import transforms

from src.utils.metrics import (
    mse,
    rmse,
    cmpa_metrics,
)
from src.utils.data_utils import CONSTANTS, WEIGHT_DICT
from src.utils.utils import get_lead_time_dict


class ModelInterface(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        compile: bool = False,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt        
        self.save_hyperparameters(ignore=['net'], logger=False)

        self.net = net
        
            
    def set_base_intervals_and_lead_times(self, train_lead_times, val_lead_times):
        # train_lead_times: list of base intervals, e.g., [6, 12, 24]
        # val_lead_times: list of target lead times, e.g., [72, 120]
        self.val_lead_times = val_lead_times
        self.train_lead_times = train_lead_times

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon
        
    def set_transforms(self, inp_transform):
        self.inp_transform = inp_transform
        self.reverse_inp_transform = self.get_reverse_transform(inp_transform)

    def get_reverse_transform(self, transform):
        mean, std = transform.mean, transform.std
        std_reverse = 1 / std
        mean_reverse = -mean * std_reverse
        return transforms.Normalize(mean_reverse, std_reverse)
    
    
    def forward(self, x: torch.Tensor, hour_of_year: torch.Tensor, lead_time: int, variables) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        norm_preds = []
        for _ in range(lead_time):
            x = self.net(x, hour_of_year)
            hour_of_year = (hour_of_year + 1) % 8784
            norm_preds.append(x)
        norm_preds = torch.stack(norm_preds, dim=1) # (B, max_T, V, H, W)
        return norm_preds
    
    def forward_train(
        self, x: torch.Tensor, hour_of_year: torch.Tensor, train_lead_times, variables
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param 
        x: initial condition, B, V, H, W.
        variables: list of variable names.
        train_lead_times: B, T, can be different across the B dimension


        :return: 
            - A list of predictions.
        """
        norm_preds = []
        max_lead_time = int(train_lead_times.max())

        norm_preds = self(x, hour_of_year, max_lead_time, variables)    # (B, Max_T, V, H, W)
        train_lead_times_idx = (train_lead_times[0] - 1).int()
        norm_preds = norm_preds[:, train_lead_times_idx]    # (B, T, V, H, W)
        # x is always in the normalized input space
        return norm_preds

    def training_step(self, batch: Any, batch_idx: int):
        x, targets, hour_of_year, train_lead_times, exist_cmpa, variables = batch
        norm_preds = self.forward_train(x, hour_of_year, train_lead_times, variables)
        norm_preds = norm_preds.flatten(0, 1)  # B*T, V, H, W
        targets = targets.flatten(0, 1)  # B*T, V, H, W
        exist_cmpa = exist_cmpa.flatten(0, 1)
        loss_dict = mse(
            norm_preds,
            targets,
            variables,
            exist_cmpa,
            weighted=True,
            weight_dict=WEIGHT_DICT
        )
        
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=x.shape[0],
                sync_dist=True,
            )

        return loss_dict[f"w_mse_aggregate"]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, dict, List[str]],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.evaluate(batch, self.val_lead_times, "val")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        pass

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.evaluate(batch, self.val_lead_times, "test")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def forward_validation(self, x: torch.Tensor, hour_of_year: torch.Tensor, lead_time: int, variables):
        # x: initial condition, B, V, H, W
        # variables: list of variable names
        # lead_time: scalar value, e.g., 168, use the same interval across the batch

        norm_preds = self(x, hour_of_year, lead_time, variables)
        return norm_preds
        
    def get_loss_dict(
        self, y: torch.Tensor, yhat: torch.Tensor, exist_cmpa: torch.Tensor, variables, list_metrics, postfix, stage
        ) -> Dict:
        """Compute the loss for a single step on a batch of data from the validation/test set.

        :param y: ground truth.
        :param yhat: prediction.
        :param stage: validation/test
        """
        all_loss_dicts = []
        for metric in list_metrics:
            loss_dict = metric(
                yhat,
                y,
                self.reverse_inp_transform,
                exist_cmpa, 
                variables,
                log_postfix=postfix,
                weighted=False,
                weight_dict=WEIGHT_DICT
            )
            all_loss_dicts.append(loss_dict)
        
        final_loss_dict = {}
        for d in all_loss_dicts:
            final_loss_dict.update(d)
            
        final_loss_dict = {f"{stage}/{k}": v for k, v in final_loss_dict.items()}
        return final_loss_dict
            
    def evaluate(
        self, batch: Tuple[torch.Tensor, torch.tensor, Dict, List[str]],
        val_lead_times: List[int],
        stage: str
    ):
        x, dict_y, hour_of_year, dict_exist_cmpa, variables = batch
        assert val_lead_times == list(dict_y.keys())
        max_lead_time = int(max(val_lead_times))
        norm_preds = self(x, hour_of_year, max_lead_time, variables)    # (B, max_T, V, H, W)
        val_lead_times_idx = torch.tensor(val_lead_times) - 1
        norm_preds = norm_preds[:, val_lead_times_idx]

        loss_per_lead_time = 0
        for idx, target_lead_time in enumerate(val_lead_times):
            # all_norm_preds = []
            norm_pred = norm_preds[:, idx]
            base_loss_dict = self.get_loss_dict(
                dict_y[target_lead_time],
                norm_pred,
                dict_exist_cmpa[target_lead_time],
                variables,
                list_metrics=[rmse, cmpa_metrics],
                postfix=f"{target_lead_time}_hrs_ensemble_mean",
                stage=stage,
            )
            loss_per_lead_time += base_loss_dict[f"{stage}/aggregate_normalized_rmse_{target_lead_time}_hrs_ensemble_mean"]
            # all_norm_preds.append(norm_pred)
            
            self.log_dict(
                base_loss_dict,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
        
        target_loss_dict = {f"{stage}/aggregate_normalized_rmse_all_ensemble_mean": loss_per_lead_time / len(val_lead_times)}
        self.log_dict(
            target_loss_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        n_steps_per_machine = len(self.trainer.datamodule.train_dataloader())
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes))

        lr_scheduler = self.hparams.scheduler(optimizer=optimizer,
                                            warmup_epochs=self.hparams.warmup_epochs * n_steps,
                                            max_epochs=self.hparams.max_epochs * n_steps,
                                            warmup_start_lr=self.hparams.warmup_start_lr,
                                            eta_min=self.hparams.eta_min
                                            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

if __name__ == "__main__":
    _ = ModelInterface(None, None, None, None)
