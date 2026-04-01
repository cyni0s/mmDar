# File for training the model behind RadarHD

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import datetime
import json
import gc
import shutil
import math

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim

import numpy as np
from torchinfo import summary
from PIL import Image
from scipy.io import savemat

from train_test_utils.dataloader import *
from train_test_utils.model import *
from train_test_utils.dice_score import dice_loss

"""
## Constants and hyperparameters
"""
params = {
    'model_name': 'baseline',
    'expt': 3,
    'batch_size': 24,
    'lr': 1.5e-4,
    'num_epochs': 400,
    'msew': 0.9,
    'dicew': 0.1,
    'optim': 'adam',
    'model_caption': 'unet 1. optimized 5090 run',
    'expt_caption': 'batch=24, lr=1.5e-4, bf16, 400 epochs — conservative Adam scaling',
    'data': 5,
    'history': 40,
    'reload': False,
    'reload_namestr': '',
    'reload_epoch': -1,
    'gpu': 1,
    'mixed_precision': True,  # bf16 forward pass, fp32 loss
    'grad_accum_steps': 1,
    'lr_schedule': 'none',
    'warmup_epochs': 0,
    'min_lr': 0.0,
}

# Prepared candidate for next 5090 run (do not auto-enable):
# params.update(params_5090_warmup_candidate)
params_5090_warmup_candidate = {
    'expt': 3,
    'batch_size': 48,
    'lr': 3e-4,  # target range: 2e-4 .. 4e-4
    'mixed_precision': True,
    'grad_accum_steps': 1,
    'lr_schedule': 'linear_warmup_cosine',
    'warmup_epochs': 10,
    'min_lr': 3e-5,
    'model_caption': 'unet 1. 5090 warmup candidate',
    'expt_caption': 'Prepared candidate: batch=48, lr=3e-4, warmup+cosine, bf16',
}

def main():
    print(torch.__version__)
    torch.manual_seed(0)

    # Can be set to cuda/cpu. Make sure model and data are moved to cuda if cuda is used
    if params['gpu'] == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dt = datetime.datetime.now()-datetime.timedelta(hours=4)
    dt = dt.strftime("%Y%m%d-%H%M%S")

    name_str = params['model_name'] + '_' + str(params['expt']) + '_' + dt

    LOG_DIR = './logs/' + name_str + '/'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, 'params.json'), 'w') as f:
        json.dump(params, f)
    train_log = os.path.join(LOG_DIR, 'train_log.txt')
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'tensorboard'))

    # Creating models
    gen = UNet1(params['history']+1,1).to(device)
    summary(gen, input_size=(1, params['history']+1, 256, 64))

    train_log_interval = 100
    model_save_interval = 10

    if params['optim'] == 'adam':
        gen_optimizer = optim.Adam(gen.parameters(), lr=params['lr'], weight_decay=0.0005)
    elif params['optim'] == 'rmsprop':
        gen_optimizer = optim.RMSprop(gen.parameters(), lr=params['lr'], weight_decay=1e-8, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {params['optim']}")

    mse_loss_fn = torch.nn.BCELoss()
    grad_accum_steps = max(1, int(params.get('grad_accum_steps', 1)))

    # LR schedule is applied per optimizer step (not per epoch)
    schedule_name = params.get('lr_schedule', 'none')
    scheduler = None
    if schedule_name == 'linear_warmup_cosine':
        updates_per_epoch = (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps
        total_updates = params['num_epochs'] * updates_per_epoch
        warmup_updates = max(0, int(params.get('warmup_epochs', 0))) * updates_per_epoch
        min_lr = float(params.get('min_lr', 0.0))
        base_lr = float(params['lr'])
        min_lr_factor = 0.0 if base_lr <= 0 else max(0.0, min(1.0, min_lr / base_lr))

        def lr_lambda(step_idx):
            if warmup_updates > 0 and step_idx < warmup_updates:
                return float(step_idx + 1) / float(max(1, warmup_updates))
            if total_updates <= warmup_updates:
                return 1.0
            progress = float(step_idx - warmup_updates) / float(max(1, total_updates - warmup_updates))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine

        scheduler = optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda)
    elif schedule_name != 'none':
        raise ValueError(f"Unsupported lr_schedule: {schedule_name}")

    # Mixed precision setup (bf16 AMP for scaled-batch runs)
    use_amp = params.get('mixed_precision', False)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    if params['reload']:
        epoch_num = '%03d' % params['reload_epoch']
        model_file = './logs/' + params['reload_namestr'] + '/' + epoch_num + '.pt_gen'
        checkpoint = torch.load(model_file, map_location=device)
        gen.load_state_dict(checkpoint['state_dict'])

    t0 = time.time()

    best_loss = float('inf')

    for epoch in range(params['num_epochs']):

        print("="*10 + "Epoch " + str(epoch) + "="*10)

        # Training -----------------------------------------------------------------------------------
        gen.train()

        losses = []
        gen_optimizer.zero_grad(set_to_none=True)

        for batch_idx, (radar, lidar) in enumerate(train_loader):
            radar = radar.to(device)
            lidar = lidar.to(device)

            # Train
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    generated_images = gen(radar)
                # BCELoss is unsafe in autocast; cast to float32 for loss computation
                generated_images_f32 = generated_images.float()
                loss1 = mse_loss_fn(generated_images_f32, lidar.float())
                loss2 = dice_loss(generated_images_f32, lidar.float())
                gen_loss = params['msew']*loss1 + params['dicew']*loss2
                scaler.scale(gen_loss / grad_accum_steps).backward()
            else:
                generated_images = gen(radar)
                loss1 = mse_loss_fn(generated_images, lidar)
                loss2 = dice_loss(generated_images, lidar)
                gen_loss = params['msew']*loss1 + params['dicew']*loss2
                (gen_loss / grad_accum_steps).backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                if use_amp:
                    scaler.step(gen_optimizer)
                    scaler.update()
                else:
                    gen_optimizer.step()
                gen_optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            losses.append(gen_loss.item())

            info = ''
            if (batch_idx % train_log_interval == 0):
                info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tGen Loss: {:.6f} '.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), gen_loss.item())

            if len(info) > 0:
                with open(train_log, 'a+') as f:
                    f.write(info + "\n")
                    print(info)

        epoch_loss = np.mean(losses)
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        writer.add_scalar('LR/current', gen_optimizer.param_groups[0]['lr'], epoch)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_checkpoint = {'state_dict': gen.state_dict(),
                               'optimizer_state_dict': gen_optimizer.state_dict()}
            torch.save(best_checkpoint, os.path.join(LOG_DIR, 'best.pt_gen'))

        if epoch % model_save_interval == 0:
            checkpoint = {'state_dict': gen.state_dict(),
                            'optimizer_state_dict': gen_optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(LOG_DIR, '%03d.pt_gen' % epoch))

        gc.collect()

    writer.close()
    t1 = time.time()
    print(t1 - t0)

# ****************************  DATALOADER ******************************
# NOTE: Dataloader is constructed at module scope (outside main()).
# This means importing this file as a module triggers data loading.
# Kept as-is for backward compatibility; wrap in if __name__ == '__main__' guard later.
# history=40 past frames + 1 current frame = 41 input channels

print('Loading data')
basepath = './dataset_' + str(params['data']) + '/'

orig_size = [256, 64, 512]
reqd_size = [256, 64, 512]

training_set = Dataset(basepath, 'train',
                        RBINS=reqd_size[0], ABINS_RADAR=reqd_size[1], ABINS_LIDAR=reqd_size[2],
                        RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1], ABINS_LIDAR_ORIG=orig_size[2],
                        M=params['history'])
train_loader = torch.utils.data.DataLoader(
    training_set, batch_size=params['batch_size'], shuffle=True,
    num_workers=4, pin_memory=True)

# ***********************************************************************

main()
