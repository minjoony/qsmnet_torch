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

from utils import *
from network import *
from train_params import *

GPU_NUM = 1
TAG = 'nrmse'
TEST_PATH = '/data/minjoon/x-sep/sub4/'
CHECKPOINT_PATH = '/home/minjoon/x-separation/ISMRM_2022/QSMadded/Checkpoint/211105_init_0001/'
RESULT_PATH = CHECKPOINT_PATH + 'Results/'
RESULT_FILE_NAME = 'qsm_Sub4predMaps_best_'
CSF_MASK_EXIST = True

### Loading CSF mask ###
if CSF_MASK_EXIST == True:
    MASK_PATH = TEST_PATH + 'csf_mask.mat'
    m = scipy.io.loadmat(MASK_PATH)
    csf_mask = m['mask']

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_set = valid_dataset(TEST_PATH)

### Network & Data loader setting ###
model = QSMnet(channel_in=CHANNEL_IN, kernel_size=KERNEL_SIZE).to(device)
load_file_name = CHECKPOINT_PATH + 'best_' + TAG + '.pth.tar'
checkpoint = torch.load(load_file_name)
model.load_state_dict(checkpoint['state_dict'])
best_epoch = checkpoint['epoch']

print(f'Best epoch: {best_epoch}\n')

print("------ Testing is started ------")
with torch.no_grad():
    model.eval()
    
    pred_qsm_map = np.zeros(test_set.matrix_size)

    valid_loss_list = []
    nrmse_list = []
    psnr_list = []
    ssim_list = []
    
    time_list = []

    for direction in range(test_set.matrix_size[-1]):
        ### Setting dataset ###
        local_f_batch = torch.tensor(test_set.field[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
        qsm_batch = torch.tensor(test_set.qsm[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
        m_batch = torch.tensor(test_set.mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
        csf_mask_batch = torch.tensor(csf_mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()

        ### Normalization ###
        local_f_batch = ((local_f_batch.cpu() - test_set.field_mean) / test_set.field_std).to(device)
        qsm_batch = ((qsm_batch.cpu() - test_set.qsm_mean) / test_set.qsm_std).to(device)
        
        ### Masking ###
        local_f_batch = local_f_batch * m_batch
        qsm_batch = qsm_batch * m_batch

        start_time = time.time()
        pred = model(local_f_batch)
        time_list.append(time.time() - start_time)
        
        l1loss = l1_loss(pred, qsm_batch)

        ### De-normalization ###
        pred_qsm = ((pred[:, 0, ...].cpu() * test_set.qsm_std) + test_set.qsm_mean).to(device).squeeze()
        label_qsm = ((qsm_batch.cpu() * test_set.qsm_std) + test_set.qsm_mean).to(device).squeeze()
        
        ### Metric calculation ###
        if CSF_MASK_EXIST == False:
            nrmse = NRMSE(pred_qsm, label_qsm, m_batch)
            psnr = PSNR(pred_qsm, label_qsm, m_batch)
            ssim = SSIM(pred_qsm.cpu(), label_qsm.cpu(), m_batch.cpu())

        elif CSF_MASK_EXIST == True:
            nrmse = NRMSE(pred_qsm, label_qsm, csf_mask_batch)
            psnr = PSNR(pred_qsm, label_qsm, csf_mask_batch)
            ssim = SSIM(pred_qsm.cpu(), label_qsm.cpu(), csf_mask_batch.cpu())

        valid_loss_list.append(l1loss.item())
        nrmse_list.append(nrmse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        
        pred_qsm_map[..., direction] = label_qsm.cpu()
        del(local_f_batch, qsm_batch, m_batch, l1loss); torch.cuda.empty_cache();
    
    test_loss = np.mean(valid_loss_list)
    NRMSE_mean = np.mean(nrmse_list)
    PSNR_mean = np.mean(psnr_list)
    SSIM_mean = np.mean(ssim_list)
    
    NRMSE_std = np.std(nrmse_list)
    PSNR_std = np.std(psnr_list)
    SSIM_std = np.std(ssim_list)
    total_time = np.mean(time_list)

    scipy.io.savemat(RESULT_PATH + RESULT_FILE_NAME + TAG + '.mat',
                     mdict={'label_qsm': test_set.qsm,
                            'label_local': test_set.field,
                            'pred_qsm': pred_qsm_map,
                            'NRMSEmean': NRMSE_mean,
                            'PSNRmean': PSNR_mean,
                            'SSIMmean': SSIM_mean,
                            'NRMSEstd': NRMSE_std,
                            'PSNRstd': PSNR_std,
                            'SSIMstd': SSIM_std})

print(f'NRMSE: {NRMSE_mean}, {NRMSE_std}  PSNR: {PSNR_mean}, {PSNR_std}  SSIM: {SSIM_mean}, {SSIM_std}')
print(f'Total inference time: {total_time}')

print("------ Testing is finished ------")