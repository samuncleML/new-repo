param(
    [string] = "./example_data",
    [string] = "my_dataset_KM_UNet",
    [int] = 256,
    [int] = 256,
    [string] = "KM_UNet"
)

python train.py --arch  --dataset my_dataset --input_w  --input_h  --name  --data_dir 
