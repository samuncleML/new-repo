import argparse
import os
import sys
import time
from collections import OrderedDict
from glob import glob
import random
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

from src import archs

from src import losses
from src.dataset import Dataset

from src.metrics import iou_score, indicators

from src.utils import AverageMeter, str2bool

try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

import shutil
import os
import subprocess

from pdb import set_trace as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='KMPUNet')

    parser.add_argument('--base_channels', default=8, type=int,
                        help='KMPUNet stem width (paper-scale ~1M params at 8)')
    parser.add_argument('--groups', default=4, type=int,
                        help='KMPUNet PCMB channel groups (paper found G=4 best)')
    parser.add_argument('--d_state', default=16, type=int,
                        help='KMPUNet Mamba branch SSM state dimension')    
    
    parser.add_argument('--resume_from', default=None, type=str,
                    help='path to checkpoint_latest.pth to resume training from')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='data', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='output dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'AdamW', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')



    config = parser.parse_args()

    return config



def train(config, train_loader, model, criterion, optimizer, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()             # scaled FP16 gradients
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

time_list = []

def main():
    seed_torch()
    config = vars(parse_args())

    # 设置数据集特定参数
    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext = '.png'  # 默认值
    if dataset_name == 'busi':
        mask_ext = '.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'
    elif dataset_name == 'isic2018':
        mask_ext = '.png'
    elif dataset_name == 'isic2017':
        mask_ext = '.png'    
    else:
        mask_ext = '.png'  # 默认值

    # 确保在使用 mask_ext 之前已经对其赋值
    config['mask_ext'] = mask_ext

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    exp_name = config['name']
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    # create model
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        base_channels=config['base_channels'],
        groups=config['groups'],
        d_state=config['d_state'],
        embed_dims=config['input_list'],   # ignored by KMPUNet, harmless to leave
        no_kan=config['no_kan']            # ignored by KMPUNet, harmless to leave
    )

    model = model.to(device)

    param_groups = []

    

    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():  # higher lr for kan layers
            param_groups.append({
                'params': param,
                'lr': config['kan_lr'],
                'weight_decay': config['kan_weight_decay']
            })
        else:
            param_groups.append({
                'params': param,
                'lr': config['lr'],
                'weight_decay': config['weight_decay']
            })

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            param_groups,
            lr=config['lr'],
            momentum=config['momentum'],
            nesterov=config['nesterov'],
            weight_decay=config['weight_decay']
        )
    else:
        raise NotImplementedError
                
        # Resume from checkpoint if provided
    start_epoch = 0
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
    ])

    if config['resume_from']:
        print(f"=> resuming from checkpoint: {config['resume_from']}")
        checkpoint = torch.load(config['resume_from'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        log = checkpoint.get('log', log)
        print(f"=> resuming at epoch {start_epoch}/{config['epochs']}")

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['min_lr'],
        last_epoch=start_epoch - 1
      )
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config['factor'],
        patience=config['patience'],
        verbose=1,
        min_lr=config['min_lr']
            )
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(e) for e in config['milestones'].split(',')],
        gamma=config['gamma'],
        last_epoch=start_epoch - 1
            )
    elif config['scheduler'] == 'ConstantLR':
         scheduler = None
    else:
        raise NotImplementedError
        

    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('src/archs.py', f'{output_dir}/{exp_name}/')

    # Data loading code
    img_dir = os.path.join(config['data_dir'], config['dataset'], 'images')
    mask_dir = os.path.join(config['data_dir'], config['dataset'], 'masks')

    img_ids = sorted(glob(os.path.join(img_dir, '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    print(f"Loaded {len(img_ids)} images from {img_dir}")
        
        
    
    train_img_ids, val_img_ids = train_test_split(
        img_ids,
        test_size=0.2,
        random_state=config['dataseed']
    )

    STAGES = [
    {'img_size': 32, 'epochs': 3, 'lr': 1e-3, 'batch_size': 64},
    {'img_size': 64, 'epochs': 3, 'lr': 7e-4, 'batch_size': 32},
    {'img_size': 128, 'epochs': 15, 'lr': 5e-4, 'batch_size': 32},
    {'img_size': 224, 'epochs': 10, 'lr': 2e-4, 'batch_size': 8 },
    ]

    counter = 0

    for stage in STAGES:
        train_transform = A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.Resize(stage['img_size'], stage['img_size']),
            A.Normalize(),
        ])

        val_transform = A.Compose([
            A.Resize(stage['img_size'], stage['img_size']),
            A.Normalize(),
        ])

        train_dataset = Dataset(
            img_ids=train_img_ids,
            img_dir=img_dir,
            mask_dir=mask_dir,
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config['num_classes'],
            transform=train_transform
        )

        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=img_dir,
            mask_dir=mask_dir,
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config['num_classes'],
            transform=val_transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=stage['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False
        )


        initial_training_time = time.perf_counter()
    
        start_epoch = 0 if stage == STAGES[0] else counter
        for epoch in range(start_epoch, stage['epochs']):

            initial = time.perf_counter()
            counter += 1
            print('Epoch [%d/%d]' % (epoch, config['epochs']))

            # train for one epoch
            train_log = train(config, train_loader, model, criterion, optimizer, device)

            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler.step()
            elif config['scheduler'] == 'MultiStepLR':
                scheduler.step()
            
            scheduler.step()
            # Note: ReduceLROnPlateau is not supported in this setup since it
            # requires per-epoch validation loss, which we no longer compute.

            print('loss %.4f - iou %.4f' % (train_log['loss'], train_log['iou']))

            # 记录当前学习率
            current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            log['epoch'].append(epoch)
            log['lr'].append(current_lrs)
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])

            pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

            my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
            my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)

            # Save a rolling checkpoint every epoch so a Colab disconnect
            # doesn't lose all progress. Overwrites the same file each time.
            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'log': log,
                }
            torch.save(checkpoint, f'{output_dir}/{exp_name}/checkpoint_latest.pth')
            print(f'=> saved checkpoint at epoch {epoch+1}')

            torch.cuda.empty_cache()
            time_per_epoch = time.perf_counter() - initial
            print(f'Time used for this epoch is {time_per_epoch}')
            time_list.append(time_per_epoch)
    
    total_training_time = initial_training_time - time.perf_counter()
    total_minutes = total_training_time/60
    print(f'Time used for all epochs is {total_minutes}')

    # evaluate on validation set once, after all training epochs
    val_log = validate(config, val_loader, model, criterion, device)
    print('Final val_loss %.4f - val_iou %.4f - val_dice %.4f' %
          (val_log['loss'], val_log['iou'], val_log['dice']))

    torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
    print("=> saved final model")

    my_writer.add_scalar('val/loss', val_log['loss'], global_step=config['epochs'])
    my_writer.add_scalar('val/iou', val_log['iou'], global_step=config['epochs'])
    my_writer.add_scalar('val/dice', val_log['dice'], global_step=config['epochs'])

    my_writer.close()


if __name__ == '__main__':
    main()
    print(time_list)
