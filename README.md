# KM-UNet training workspace

This workspace has been reorganized into a cleaner training layout:

- src/: model, dataset, loss, metric, and utility modules
- data/busi/: training-ready BUSI images and masks
- models/: model checkpoints and saved weights
- outputs/: training logs, configs, and TensorBoard runs
- scripts/: dataset preparation helpers

## Quick start

1. Prepare the BUSI data layout:
   python scripts/prepare_busi_data.py

2. Train:
   python train.py --dataset busi --data_dir data --output_dir outputs --name busi_km_unet

## Notes

- The training script now defaults to the reorganized data directory.
- The dataset loader expects images in data/busi/images and masks in data/busi/masks/0.
