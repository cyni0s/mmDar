# File for training the model behind RadarHD

import time
import os
import datetime
import json
import gc
import shutil

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
    'model_name': '13',
    'expt': 1,
    'batch_size': 6,
    'lr': 1e-4,
    'num_epochs': 200,
    'msew': 0.9,
    'dicew': 0.1,
    'optim': 'adam',
    'model_caption': 'unet 1.',
    'expt_caption': '',
    'data': 5,
    'history': 40,
    'reload': False,
    'reload_namestr': '',
    'reload_epoch': -1,
    'gpu': 1,
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
    summary(gen, (params['history']+1, 256, 64))

    train_log_interval = 100
    model_save_interval = 10
    
    if params['optim'] == 'adam':
        gen_optimizer = optim.Adam(gen.parameters(), lr=params['lr'], weight_decay=0.0005)
    elif params['optim'] == 'rmsprop':
        gen_optimizer = optim.RMSprop(gen.parameters(), lr=params['lr'], weight_decay=1e-8, momentum=0.9)

    mse_loss_fn = torch.nn.BCELoss()

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
        
        for batch_idx, (radar, lidar) in enumerate(train_loader):
            radar = radar.to(device)
            lidar = lidar.to(device)
            batch_size = radar.size(0)

            # Train
            gen_optimizer.zero_grad()
            generated_images = gen(radar)

            loss1 = mse_loss_fn(generated_images, lidar)
            loss2 = dice_loss(generated_images, lidar)
            gen_loss = params['msew']*loss1 + params['dicew']*loss2

            gen_loss.backward()
            gen_optimizer.step()
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
train_loader = torch.utils.data.DataLoader(training_set, batch_size=params['batch_size'], shuffle=True)

# ***********************************************************************

main()
