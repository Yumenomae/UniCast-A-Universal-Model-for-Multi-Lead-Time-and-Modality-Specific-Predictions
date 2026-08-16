## Usage

### Download and process data

We convert the ctl file to H5DF format for easier data loading with Pytorch. To do this, run

```bash
python src/data_preprocessing/process_one_step_data.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
    --ctl_path [CTL_PATH] \
    --start_year [START_YEAR] \
    --end_year [END_YEAR] \
    --split [SPLIT] \
```

```bash
python src/data_preprocessing/process_one_step_data.py \
    --root_dir '/home-ssd/Users/gm_lhy' \
    --save_dir './data' \
    --ctl_path './post.ctl' \
    --start_year 2023 \
    --end_year 2023 \
    --split 'test' \
```

We check whether there is nan value in data. To do this, run

```bash
python src/data_preprocessing/check_nan.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
```

```bash
python src/data_preprocessing/check_nan.py \
    --root_dir './data/val' \
    --save_dir './data' \
```

We pre-compute the normalization constants for training. To do this, run

```bash
python src/data_preprocessing/compute_normalization.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
```

```bash
python src/data_preprocessing/compute_normalization.py \
    --root_dir '/home-ssd/Users/gm_lhy/zhengjq/Ours/data/train' \
    --save_dir './data' \
```

We pre-compute the time-position embedding for training. To do this, run

```bash
python src/data_preprocessing/process_one_step_embed.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
```

```bash
python src/data_preprocessing/process_one_step_embed.py \
    --root_dir './data' \
    --save_dir './data/embed' \
```

## Workflow

**Basic workflow**

1. Write your PyTorch Lightning module (see [models/mnist_module.py](src/models/mnist_module.py) for example)
2. Write your PyTorch Lightning datamodule (see [data/mnist_datamodule.py](src/data/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to model and datamodule
4. Run training with chosen experiment config:


**Testing**
   ```bash
   python src/train.py experiment=Control_Test.yaml
   ```

**Himmel Pilot**
   ```bash
   # The data is set to 1/10 
   python src/train.py experiment=Himmel_pilot_4M.yaml  # 2025-03-31_10-44-21 

   python src/train.py experiment=Himmel_pilot_16M.yaml # 2025-03-31_10-46-35

   python src/train.py experiment=Himmel_pilot_32M.yaml # 2025-03-31_10-46-35 

   python src/visual.py experiment=Himmel_40M.yaml ckpt_path='./logs/train/runs/Himmel_40M/checkpoints/epoch_022.ckpt'

   ```

**Himmel**
   ```bash
   python src/train.py experiment=Himmel_4M.yaml  # 2025-03-31_11-37-55 gn002 down

   python src/train.py experiment=Himmel_8M.yaml ckpt_path='./logs/train/runs/2025-04-03_09-28-13/checkpoints/last.ckpt' # 2025-04-07_09-16-24 gn004 
   # past: 2025-04-03_09-28-13

   python src/train.py experiment=Himmel_16M.yaml # 2025-03-31_11-43-49 gn005 

   python src/train.py experiment=Himmel_40M.yaml ckpt_path='./logs/train/runs/2025-03-31_12-31-56/checkpoints/epoch_011.ckpt' # 2025-04-01_13-55-49 finished

   python src/train.py experiment=Himmel_64M.yaml  # 2025-04-11_09-37-28 gn002 

   ```