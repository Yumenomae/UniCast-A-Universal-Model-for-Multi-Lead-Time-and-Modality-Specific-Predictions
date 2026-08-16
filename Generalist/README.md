## Usage

### Download and process data

We convert the ctl file to H5DF format for easier data loading with Pytorch. To do this, run

```bash
python src/data_preprocessing/process_one_step_data.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
    --ctl_path [CTL_PATH] \ # Our data is in .ctl format
    --start_year [START_YEAR] \
    --end_year [END_YEAR] \
    --split [SPLIT] \
```


We check whether there is nan value in data. To do this, run

```bash
python src/data_preprocessing/check_nan.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
```


We pre-compute the normalization constants for training. To do this, run

```bash
python src/data_preprocessing/compute_normalization.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
```


We pre-compute the time-position embedding for training. To do this, run

```bash
python src/data_preprocessing/process_one_step_embed.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
```

## Workflow

**Basic workflow**

1. Write your PyTorch Lightning module (see [models/mnist_module.py](src/models/mnist_module.py) for example)
2. Write your PyTorch Lightning datamodule (see [data/mnist_datamodule.py](src/data/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to model and datamodule
4. Run training with chosen experiment config.

**Training UniCast's Generalist**
   ```bash
srun python src/train.py \
        experiment=UniCast.yaml \
        model._target_=src.models.UniCast_module.ModelInterface \
        data=LimitArea_hourly \
        data.train_lead_times=[1,3,6,12] \
        data.val_lead_times=[1,3,6,12,24,48] \
   ```

**Training UniCast_star's Generalist**
   ```bash
srun python src/train.py \
        experiment=UniCast.yaml \
        model._target_=src.models.UniCast_module.ModelInterface \
        data=LimitArea_0_6_12_18 \
        data.train_lead_times=[1,3,6,12] \
        data.val_lead_times=[1,3,6,12,24,48] \
   ```

**Testing UniCast's Generalist**
   ```bash
srun python src/eval.py \
        experiment=UniCast.yaml \
        model._target_=src.models.UniCast_module.ModelInterface \
        data=LimitArea_hourly \
        ckpt_path=YOUR_CHECKPOINT_PATH \
   ```

**Testing UniCast_star's Generalist**
   ```bash
srun python src/eval.py \
        experiment=UniCast.yaml \
        model._target_=src.models.UniCast_module.ModelInterface \
        data=LimitArea_0_6_12_18 \
        ckpt_path=YOUR_CHECKPOINT_PATH \
   ```
