This folder is a minimal training package for KM-UNet.

Expected dataset layout:
- images/0001.png
- masks/0/0001.png

Training command example:
python train.py --arch KM_UNet --dataset my_dataset --input_w 256 --input_h 256 --name my_dataset_KM_UNet --data_dir ./example_data
