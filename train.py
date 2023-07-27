'''
#
# Description:
#  Training code of qsmnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 23.07.27
'''
import os
import logging
import glob
import time
import math
import shutil
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import logging_helper as logging_helper
from utils import *
from network import *
from custom_dataset import *
from train_params import parse

args = parse()
writer = SummaryWriter(args.CHECKPOINT_PATH + 'runs/')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= args.GPU_NUM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

createDirectory(args.CHECKPOINT_PATH)
createDirectory(args.CHECKPOINT_PATH+ 'Results')

### Logger setting ###
logger = logging.getLogger("module.train")
logger.setLevel(logging.INFO)
logging_helper.setup(args.CHECKPOINT_PATH + 'Results','log.txt')

nowDate = datetime.datetime.now().strftime('%Y-%m-%d')
nowTime = datetime.datetime.now().strftime('%H:%M:%S')
logger.info(f'Date: {nowDate}  {nowTime}')

for key, value in vars(args).items():
    logger.info('{:15s}: {}'.format(key,value))

### Random seed ###
os.environ['PYTHONHASHargs.SEED'] = str()
random.seed(args.SEED)
np.random.seed(args.SEED)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
torch.cuda.manual_seed_all(args.SEED)
torch.random.manual_seed(args.SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

### Network & Data loader setting ###
model = QSMnet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)

if torch.cuda.device_count() > 1:
    logger.info(f'Multi GPU - num: {torch.cuda.device_count()} - are used')
    model = nn.DataParallel(model).to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=args.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.98, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.LR_EXP_DECAY_GAMMA)

train_set = train_dataset(args)
valid_set = valid_dataset(args)

train_loader = DataLoader(train_set, batch_size = args.BATCH_SIZE, shuffle = True, num_workers = 8)

dipole = dipole_kernel((64,64,64), voxel_size=(1,1,1), B0_dir=(0,0,1))

logger.info(f'Num of batches: {len(train_set)}')

step = 0
train_loss = []
valid_loss = []
nrmse = []
psnr = []
ssim = []
best_loss = math.inf; best_nrmse = math.inf; best_psnr = -math.inf; best_ssim = -math.inf
best_epoch_loss = 0; best_epoch_nrmse = 0; best_epoch_psnr = 0; best_epoch_ssim = 0
data_index = 0; local_f_index = 1; qsm_index = 2; mask_index = 2;

start_time = time.time()

logger.info("------ Training is started ------")
for epoch in tqdm(range(args.TRAIN_EPOCH)):
    ### Training ###
    epoch_time = time.time()
    
    train_loss_list = []
    train_mdloss_list = []
    train_gdloss_list = []
    valid_loss_list =[]
    nrmse_list = []
    psnr_list = []
    ssim_list = []
    
    for train_data in tqdm(train_loader):
        model.train()
        
        index = train_data[0]
        local_f_batch = train_data[1].to(device)
        qsm_batch = train_data[2].to(device)
        m_batch = train_data[3].to(device)
        
        ### Masking ###
        # input dim: [batch_size, 1, 64, 64, 64]
        # label dim: [batch_size, 1, 64, 64, 64]
        local_f_batch = local_f_batch * m_batch
        
        pred = model(local_f_batch)
        
        ##
#         pred = pred * m_batch
#         label = qsm_batch * m_batch
        
#         pred = (pred.cpu().detach().numpy() * train_set.qsm_std) + train_set.qsm_mean
        
#         label = (label.cpu().detach().numpy() * train_set.qsm_std) + train_set.qsm_mean
        
#         scipy.io.savemat(args.CHECKPOINT_PATH + 'Results/predMaps_train.mat',
#                  mdict={'label_qsm': label,
#                         'pred_qsm': pred})
        ##
        
        loss, l1loss, mdloss, gdloss = total_loss(pred, qsm_batch, local_f_batch, m_batch, dipole, args.W_L1LOSS, args.W_MDLOSS, args.W_GDLOSS,
                                                  train_set.qsm_mean, train_set.qsm_std, train_set.field_mean, train_set.field_std)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        
        train_loss_list.append(loss.item())
        train_mdloss_list.append(mdloss.item())
        train_gdloss_list.append(gdloss.item())

        del(local_f_batch, qsm_batch, m_batch, loss, l1loss, mdloss, gdloss); torch.cuda.empty_cache();

    logger.info("Train: EPOCH %04d / %04d | LOSS %.6f | M_LOSS %.6f | G_LOSS %.6f | TIME %.1fsec | LR %.8f"
          %(epoch+1, args.TRAIN_EPOCH, np.mean(train_loss_list), np.mean(train_mdloss_list), np.mean(train_gdloss_list), time.time() - epoch_time, optimizer.param_groups[0]['lr']))
    
    ### Validation ###
    model.eval()
    
    with torch.no_grad():
        for direction in range(valid_set.matrix_size[-1]):
            local_f_batch = torch.tensor(valid_set.field[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
            qsm_batch = torch.tensor(valid_set.qsm[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
            m_batch = torch.tensor(valid_set.mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)

            ### Normalization ###
            local_f_batch = ((local_f_batch - valid_set.field_mean) / valid_set.field_std).to(device)
            qsm_batch = ((qsm_batch - valid_set.qsm_mean) / valid_set.qsm_std).to(device)

            ### Masking ###
            local_f_batch = local_f_batch * m_batch
            qsm_batch = qsm_batch * m_batch

            pred = model(local_f_batch)
            
            pred = pred * m_batch
            qsm_batch = qsm_batch * m_batch

            l1loss = l1_loss(pred, qsm_batch)
            
            ### De-normalization ###
            pred = ((pred * valid_set.qsm_std) + valid_set.qsm_mean).to(device)
            qsm_batch = ((qsm_batch * valid_set.qsm_std) + valid_set.qsm_mean).to(device)
            
            _nrmse = NRMSE(pred, qsm_batch, m_batch)
            _psnr = PSNR(pred, qsm_batch, m_batch)
            _ssim = SSIM(pred, qsm_batch, m_batch)

            valid_loss_list.append(l1loss.item())
            nrmse_list.append(_nrmse)
            psnr_list.append(_psnr)
            ssim_list.append(_ssim)
            
            del(local_f_batch, qsm_batch, m_batch, l1loss); torch.cuda.empty_cache();
        logger.info("Valid: EPOCH %04d / %04d | LOSS %.6f | NRMSE %.4f | PSNR %.4f | SSIM %.4f\n"
              %(epoch+1, args.TRAIN_EPOCH, np.mean(valid_loss_list), np.mean(_nrmse), np.mean(_psnr), np.mean(_ssim)))
        
        train_loss.append(np.mean(train_loss_list))
        valid_loss.append(np.mean(valid_loss_list))
        nrmse.append(np.mean(nrmse_list))
        psnr.append(np.mean(psnr_list))
        ssim.append(np.mean(ssim_list))
        
        writer.add_scalar("Train loss/epoch", np.mean(train_loss_list), epoch+1)      
        writer.add_scalar("valid loss/epoch", np.mean(valid_loss_list), epoch+1)
        writer.add_scalar("valid nrmse/epoch", np.mean(_nrmse), epoch+1)
        writer.add_scalar("valid psnr/epoch", np.mean(_psnr), epoch+1)
        writer.add_scalar("valid ssim/epoch", np.mean(_ssim), epoch+1)

        if np.mean(valid_loss_list) < best_loss:
            save_model(epoch+1, model, args.CHECKPOINT_PATH, 'best_loss')
            best_loss = np.mean(valid_loss_list)
            best_epoch_loss = epoch+1
        if np.mean(_nrmse) < best_nrmse:
            save_model(epoch+1, model, args.CHECKPOINT_PATH, 'best_nrmse')
            best_nrmse = np.mean(_nrmse)
            best_epoch_nrmse = epoch+1
        if np.mean(_psnr) > best_psnr:
            save_model(epoch+1, model, args.CHECKPOINT_PATH, 'best_psnr')
            best_psnr = np.mean(_psnr)
            best_epoch_psnr = epoch+1
        if np.mean(_ssim) > best_ssim:
            save_model(epoch+1, model, args.CHECKPOINT_PATH, 'best_ssim')
            best_ssim = np.mean(_ssim)
            best_epoch_ssim = epoch+1

    ### Saving the model ###
#     if (epoch+1) % args.SAVE_STEP == 0:
#         save_model(epoch+1, model, args.CHECKPOINT_PATH, epoch+1)

logger.info("------ Training is finished ------")
logger.info(f'[best epochs]\nLoss: {best_epoch_loss}\nNRMSE: {best_epoch_nrmse}\nPSNR: {best_epoch_psnr}')
logger.info(f'Total training time: {time.time() - start_time}')

### Plotting learning curve & result curves ###
epoch_list = range(1, args.TRAIN_EPOCH + 1)
plt.ylim((0.01, 0.50))
plt.plot(epoch_list, np.array(train_loss), 'y')
plt.title('Train loss Graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(args.CHECKPOINT_PATH + "Results/train_loss_graph.png")
plt.clf()

plt.ylim((0.01, 0.50))
plt.plot(epoch_list, np.array(valid_loss), 'c')
plt.title('Valid loss Graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(args.CHECKPOINT_PATH + "Results/valid_loss_graph.png")
plt.clf()

plt.plot(epoch_list, np.array(nrmse), 'y')
plt.ylim((30, 55))
plt.title('NRMSE Graph')
plt.xlabel('epoch')
plt.ylabel('NRMSE')
plt.savefig(args.CHECKPOINT_PATH + "Results/NRMSE_graph.png")
plt.clf()

plt.plot(epoch_list, np.array(psnr), 'y')
plt.ylim((40, 50))
plt.title('PSNR Graph')
plt.xlabel('epoch')
plt.ylabel('PSNR')
plt.savefig(args.CHECKPOINT_PATH + "Results/PSNR_graph.png")
plt.clf()