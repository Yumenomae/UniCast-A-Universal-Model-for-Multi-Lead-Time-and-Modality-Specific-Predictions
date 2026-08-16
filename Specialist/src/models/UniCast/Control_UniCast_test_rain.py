from typing import Any, Dict, Tuple, Dict, List
import torch
import numpy as np
from lightning import LightningModule
from torchvision.transforms import transforms

# from stormer.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import (
    cmpa_metrics_cf,
)
from src.utils.data_utils import CONSTANTS, WEIGHT_DICT, DEFAULT_PRESSURE_LEVELS
from src.utils.utils import get_lead_time_dict
from src.utils.physics_components.air_column_vars_1000hPa import compute_air_column_variable


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
        ControlledNet: torch.nn.Module,
        ControlledNet_ckpt_path: str,
        ControlNet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        compile: bool = False,
        alpha: float = 1.0,
        gamma: float = 1.0,
        gradient_scale: float = 48.28,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt        
        self.save_hyperparameters(ignore=['ControlledNet', 'ControlledNet_ckpt_path', 'ControlNet'], logger=False)

        self.ControlledNet = ControlledNet
        self.ControlNet = ControlNet

        self.load_pretrained_weights_ControlledNet(ControlledNet_ckpt_path) 
        for p in self.ControlledNet.parameters():
            p.requires_grad = False
        
    #     if pretrained_path is not None:
    #         self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights_ControlledNet(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        state_dict = checkpoint["state_dict"]
        state_dict = {k[4:]: v for k, v in state_dict.items()}
        msg = self.ControlledNet.load_state_dict(state_dict)
        print(msg)
            
    def set_base_intervals_and_lead_times(self, train_lead_times, val_lead_times):
        # train_lead_times: list of base intervals, e.g., [6, 12, 24]
        # val_lead_times: list of target lead times, e.g., [72, 120]
        self.val_lead_times = val_lead_times
        self.train_lead_times = train_lead_times

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.assist_compute = compute_air_column_variable(pressure_level=DEFAULT_PRESSURE_LEVELS,
                                                          lat=lat,
                                                          lon=lon,
                                                          a=self.hparams.alpha)
        
    def set_transforms(self, inp_transform, assist_transform):
        self.inp_transform = inp_transform
        self.reverse_inp_transform = self.get_reverse_transform(inp_transform)
        self.assist_transform = assist_transform

    def get_reverse_transform(self, transform):
        mean, std = transform.mean, transform.std
        std_reverse = 1 / std
        mean_reverse = -mean * std_reverse
        return transforms.Normalize(mean_reverse, std_reverse)
    
    # def replace_constant(self, yhat, out_variables):
    #     for i in range(yhat.shape[1]):
    #         if out_variables[i] in CONSTANTS:
    #             yhat[:, i] = 0.0
    #     return yhat
    
    def pad(self, x: torch.Tensor):
        h, w = x.shape[-2], x.shape[-1]
        if h % self.ControlledNet.patch_size == 0 and w % self.ControlledNet.patch_size == 0:
            padded_x = x
            pad_size = 0
            return padded_x, pad_size
        
        # Calculate the pad size for the height if it's not divisible by the patch size
        if h % self.ControlledNet.patch_size != 0:
            pad_size = (self.ControlledNet.patch_size - h % self.ControlledNet.patch_size) // 2
        else: pad_size = 0

        padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, pad_size), 'constant', 0)
        return padded_x, pad_size
    
    def forward(self, x: torch.Tensor, hour_of_year: torch.Tensor, lead_time, variables) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        padded_x, pad_size = self.pad(x)

        cond_x_2d = padded_x[:, :8, :, :]
        cond_x_3d = padded_x[:, 8:, :, :].reshape(x.shape[0], 5, 13, padded_x.shape[2], padded_x.shape[3])   # (B, 5, 13, 441, 845)

        x, lead_times_emb, data_shape = self.ControlledNet.first_stage_forward(padded_x, hour_of_year, lead_time, variables)
        B, C, Pl, Lat, Lon = data_shape
        cond_x, cond_lead_times_emb = self.ControlNet.first_stage_forward(cond_x_2d, cond_x_3d, hour_of_year, lead_time)

        cond_x = cond_x + x
        cond_lead_times_emb = cond_lead_times_emb + lead_times_emb
        
        zero_convs_layer3_output, zero_convs_layer4_output = self.ControlNet.middle_stage_forward(cond_x, cond_lead_times_emb)

        x = self.ControlledNet.layer1(x, lead_times_emb)
        skip = x
        x, lead_times_emb = self.ControlledNet.downsample(x, lead_times_emb)
        x = self.ControlledNet.layer2(x, lead_times_emb)

        for idx, blk in enumerate(self.ControlledNet.layer3.blocks):
            x = blk(x, lead_times_emb)
            x = x + zero_convs_layer3_output[::-1][idx]

        x, lead_times_emb = self.ControlledNet.upsample(x, lead_times_emb)

        for idx, blk in enumerate(self.ControlledNet.layer4.blocks):
            x = blk(x, lead_times_emb)
            x = x + zero_convs_layer4_output[::-1][idx]

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]

        output_surface = self.ControlledNet.patchrecovery2d(output_surface)
        output_upper_air = self.ControlledNet.patchrecovery3d(output_upper_air)

        pred = torch.concat([output_surface, output_upper_air.flatten(1,2)], dim=1)
        
        return pred[:, :, pad_size:-pad_size]
    
    def forward_wo_control(self, x: torch.Tensor, hour_of_year: torch.Tensor, lead_time, variables) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        padded_x, pad_size = self.pad(x)
        pred = self.ControlledNet(padded_x, hour_of_year, lead_time, variables)

        return pred[:, :, pad_size:-pad_size]

    def forward_wo_control_val(self, x: torch.Tensor, hour_of_year: torch.Tensor, lead_time, variables):
        # x: initial condition, B, V, H, W
        # variables: list of variable names
        # lead_time: scalar value, e.g., 168, use the same interval across the batch

        # x is always in the normalized input space
        assert isinstance(lead_time, int)
        base_lead_time = [1, 3, 6, 12]
        lead_time_step = get_lead_time_dict(base=base_lead_time, target=lead_time)

        for sub_lead_time in base_lead_time:
            # sub_lead_time = torch.Tensor([sub_lead_time]).to(device=x.device, dtype=x.dtype)
            steps = lead_time_step[sub_lead_time]
            sub_lead_time = torch.Tensor([sub_lead_time])
            sub_lead_time = sub_lead_time.repeat(x.shape[0]).to(device=x.device, dtype=x.dtype)
            for _ in range(steps):
                norm_pred_diff = self.forward_wo_control(x, hour_of_year, sub_lead_time, variables) # diff in the normalized space
                x = x + norm_pred_diff # prediction in the normalized space
                hour_of_year = (hour_of_year + sub_lead_time) % 8784
        return x
    
    def forward_assist(
        self, x: torch.Tensor, hour_of_year, variables
    ) -> torch.Tensor:
        upper_vars_dict, surface_vars_dict = self.assist_compute.pressure_level_reshape_denorm(x, self.reverse_inp_transform)
        u, v, Qv, t = upper_vars_dict['u'], upper_vars_dict['v'], upper_vars_dict['Qv'], upper_vars_dict['t']
        u10m, v10m = surface_vars_dict['u10m'], surface_vars_dict['v10m']
        q2m, ps, t2m = surface_vars_dict['q2m'], surface_vars_dict['ps'], surface_vars_dict['t2m']

        pred_x = self.forward_one_step_validation(x, hour_of_year, 1, variables)
        pred_x = self.reverse_inp_transform(pred_x)
        pred_R = pred_x[:, 0]

        pred_ac_h = self.assist_compute.predict_air_column_humidity(Qv, u, v, u10m, v10m, q2m, ps, t, t2m, pred_R)

        return pred_ac_h.unsqueeze(1) # (B, 1, H, W)

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
        norm_preds, assist_preds = [], []
        for lead_time in train_lead_times[0]:
            norm_pred = self.forward_one_step_validation(x, hour_of_year, int(lead_time), variables)

            pred_x = self.forward_wo_control_val(x, hour_of_year, int(lead_time), variables)
            assist_pred = self.forward_assist(pred_x, (hour_of_year+lead_time)%8784, variables)
            norm_preds.append(norm_pred)
            assist_preds.append(assist_pred)

        return norm_preds, assist_preds

    def training_step(self, batch: Any, batch_idx: int):
        x, targets, hour_of_year, train_lead_times, exist_cmpa, variables, assists, assists_exist_cmpa = batch
        norm_preds, assist_preds = self.forward_train(x, hour_of_year, train_lead_times, variables)
        norm_preds = torch.stack(norm_preds, dim=1).flatten(0, 1)  # B*T, V, H, W
        targets = targets.flatten(0, 1)  # B*T, V, H, W
        exist_cmpa = exist_cmpa.flatten(0, 1)

        assist_preds = torch.stack(assist_preds, dim=1).flatten(0, 1)  # B*T, 2, H, W
        assist_preds = self.assist_transform(assist_preds)
        assists = assists.flatten(0, 1) # B*T, V, H, W

        assists = self.reverse_inp_transform(assists)
        fute_R = assists[:, 0]

        upper_target_dict, surface_target_dict = self.assist_compute.pressure_level_reshape_denorm(targets, self.reverse_inp_transform)
        u, v, Qv, t = upper_target_dict['u'], upper_target_dict['v'], upper_target_dict['Qv'], upper_target_dict['t']
        u10m, v10m = surface_target_dict['u10m'], surface_target_dict['v10m']
        q2m, ps, t2m = surface_target_dict['q2m'], surface_target_dict['ps'], surface_target_dict['t2m']

        target_ac_h = self.assist_compute.predict_air_column_humidity(Qv, u, v, u10m, v10m, q2m, ps, t, t2m, fute_R)

        target_assist = target_ac_h.unsqueeze(1)
        target_assist = self.assist_transform(target_assist)
        assists_exist_cmpa = assists_exist_cmpa.flatten(0, 1)
        
        direct_loss = self.ACSLoss(norm_preds, targets, exist_cmpa)  # CMPA only
        self.log(
            "train/direct_loss",
            direct_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
            sync_dist=True
        )

        autoreg_loss = self.ACSLoss(assist_preds, target_assist, assists_exist_cmpa)
        self.log(
            "train/autoreg_loss",
            autoreg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
            sync_dist=True
        )

        # return PhyAst_fourier_loss
        return self.hparams.gamma * direct_loss + (1-self.hparams.gamma) * self.hparams.gradient_scale * autoreg_loss

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

    def forward_one_step_validation(self, x: torch.Tensor, hour_of_year: torch.Tensor, lead_time, variables):
        # x: initial condition, B, V, H, W
        # variables: list of variable names
        # lead_time: scalar value, e.g., 168, use the same interval across the batch

        # x is always in the normalized input space
        assert isinstance(lead_time, int)
        assert (lead_time in [1, 3])

        lead_time = torch.Tensor([lead_time])
        lead_time = lead_time.repeat(x.shape[0]).to(device=x.device, dtype=x.dtype)
        norm_pred_diff = self(x, hour_of_year, lead_time, variables) # diff in the normalized space
        # norm_pred_diff = self.forward_wo_control(x, hour_of_year, lead_time, variables) # diff in the normalized space
        x = x + norm_pred_diff # prediction in the normalized space
        return x

    def forward_multi_step_validation(self, x: torch.Tensor, hour_of_year: torch.Tensor, lead_time, one_step_lead_time, variables):
        # x: initial condition, B, V, H, W
        # variables: list of variable names
        # lead_time: scalar value, e.g., 168, use the same interval across the batch

        # x is always in the normalized input space
        assert isinstance(lead_time, int)
        assert isinstance(one_step_lead_time, int)
        assert (one_step_lead_time < lead_time)
        assert (one_step_lead_time in [1, 3])

        prepare_lead_time = lead_time - one_step_lead_time
        x = self.forward_wo_control_val(x, hour_of_year, prepare_lead_time, variables)
        hour_of_year = (hour_of_year + prepare_lead_time) % 8784

        x = self.forward_one_step_validation(x, hour_of_year, one_step_lead_time, variables)
        return x
        
    def get_loss_dict(
        self, y: torch.Tensor, yhat: torch.Tensor, exist_cmpa: torch.Tensor, variables, 
        transform, list_metrics, postfix, stage
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
                transform,
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
        x, dict_y, hour_of_year, dict_exist_cmpa, variables, dict_assists, dict_assists_exist_cmpa = batch
        
        for target_lead_time in val_lead_times:
            # all_norm_preds = []
            if target_lead_time in [1, 3]:
                norm_pred = self.forward_one_step_validation(x, hour_of_year, target_lead_time, variables)
            elif target_lead_time in [2]:
                norm_pred_1_step = self.forward_multi_step_validation(x, hour_of_year, target_lead_time, 1, variables)
            else:
                norm_pred_1_step = self.forward_multi_step_validation(x, hour_of_year, target_lead_time, 1, variables)
                norm_pred_3_step = self.forward_multi_step_validation(x, hour_of_year, target_lead_time, 3, variables)

            if target_lead_time in [1, 3]:
                base_loss_dict = self.get_loss_dict(
                    dict_y[target_lead_time],
                    norm_pred,
                    dict_exist_cmpa[target_lead_time],
                    variables,
                    self.reverse_inp_transform,
                    list_metrics=[cmpa_metrics_cf],
                    postfix=f"{target_lead_time}_hrs_ensemble_mean",
                    stage=stage,
                )

                self.log_dict(
                    base_loss_dict,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=x.shape[0],
                )

            elif target_lead_time in [2]:
                base_loss_dict_1_step = self.get_loss_dict(
                    dict_y[target_lead_time],
                    norm_pred_1_step,
                    dict_exist_cmpa[target_lead_time],
                    variables,
                    self.reverse_inp_transform,
                    list_metrics=[cmpa_metrics_cf],
                    postfix=f"{target_lead_time}_hrs_ensemble_mean_1_step",
                    stage=stage,
                )

                self.log_dict(
                    base_loss_dict_1_step,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=x.shape[0],
                )

            else:
                base_loss_dict_1_step = self.get_loss_dict(
                    dict_y[target_lead_time],
                    norm_pred_1_step,
                    dict_exist_cmpa[target_lead_time],
                    variables,
                    self.reverse_inp_transform,
                    list_metrics=[cmpa_metrics_cf],
                    postfix=f"{target_lead_time}_hrs_ensemble_mean_1_step",
                    stage=stage,
                )

                base_loss_dict_3_step = self.get_loss_dict(
                    dict_y[target_lead_time],
                    norm_pred_3_step,
                    dict_exist_cmpa[target_lead_time],
                    variables,
                    self.reverse_inp_transform,
                    list_metrics=[cmpa_metrics_cf],
                    postfix=f"{target_lead_time}_hrs_ensemble_mean_3_step",
                    stage=stage,
                )

                self.log_dict(
                    base_loss_dict_1_step,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=x.shape[0],
                )

                self.log_dict(
                    base_loss_dict_3_step,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=x.shape[0],
                )

            # loss_per_lead_time += base_loss_dict[f'{stage}/FCL_precp_{target_lead_time}_hrs_ensemble_mean'] + \
            #                         base_loss_dict[f'{stage}/FAL_precp_{target_lead_time}_hrs_ensemble_mean']

        
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
        # decay = []
        # no_decay = []
        # for name, m in self.named_parameters():
        #     if "channel_embed" in name or "pos_embed" in name:
        #         no_decay.append(m)
        #     else:
        #         decay.append(m)

        # optimizer = torch.optim.AdamW(
        #     [
        #         {
        #             "params": decay,
        #             "lr": self.hparams.lr,
        #             "betas": (self.hparams.beta_1, self.hparams.beta_2),
        #             "weight_decay": self.hparams.weight_decay,
        #         },
        #         {
        #             "params": no_decay,
        #             "lr": self.hparams.lr,
        #             "betas": (self.hparams.beta_1, self.hparams.beta_2),
        #             "weight_decay": 0,
        #         },
        #     ]
        # )

        # n_steps_per_machine = len(self.trainer.datamodule.train_dataloader())
        # n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes))
        # lr_scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     self.hparams.warmup_epochs * n_steps,
        #     self.hparams.max_epochs * n_steps,
        #     self.hparams.warmup_start_lr,
        #     self.hparams.eta_min,
        # )
        # scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}

if __name__ == "__main__":
    _ = ModelInterface(None, None, None, None)
