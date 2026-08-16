## Usage

### Download and process data

We pre-compute the normalization of VIWV for training. To do this, run

```bash
python src/data_preprocessing/compute_normalization_fromPs.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
```

## Workflow

**Basic workflow**

1. Write your PyTorch Lightning module (see [models/mnist_module.py](src/models/mnist_module.py) for example)
2. Write your PyTorch Lightning datamodule (see [data/mnist_datamodule.py](src/data/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to model and datamodule
4. Run training with chosen experiment config:


**Training UniCast's Specialist**
   ```bash
srun python src/train.py \
        experiment=UniCast.yaml \
        model.optimizer.lr=5e-4 \
        data=LimitArea_hourly \
        model.alpha=0.79 \
        model.gamma=0.95 \
        +model.gradient_scale=15 \
        model._target_=src.models.UniCast.Control_UniCast_wPhy_ACSLoss.ModelInterface \
        model.ControlledNet_ckpt_path='GENERALIST_PATH' \
   ```

**Training UniCast_star's Specialist**
   ```bash
srun python src/train.py \
        experiment=UniCast.yaml \
        model.optimizer.lr=5e-4 \
        data=LimitArea_0_6_12_18 \
        model.alpha=0.79 \
        model.gamma=0.95 \
        model.gradient_scale=15 \
        model._target_=src.models.UniCast.Control_UniCast_wPhy_ACSLoss.ModelInterface \
        model.ControlledNet_ckpt_path='GENERALIST_PATH' \
   ```

**Testing UniCast's Specialist**
   ```bash
srun python src/eval.py \
        experiment=UniCast.yaml \
        data=LimitArea_hourly \
        model.alpha=0.79 \
        model._target_=src.models.UniCast.Control_UniCast_test.ModelInterface \
        data.val_lead_times=[1,3] \
        ckpt_path='YOUR_CHECKPOINT_PATH' \
   ```

**Testing UniCast_star's Specialist**
   ```bash
srun python src/eval.py \
        experiment=UniCast.yaml \
        data=LimitArea_0_6_12_18 \
        model._target_=src.models.UniCast.Control_UniCast_test.ModelInterface \
        data.val_lead_times=[1,3] \
        ckpt_path='YOUR_CHECKPOINT_PATH' \
   ```