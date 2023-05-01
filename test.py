'''
#
# Description:
#  Test code of qsmnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 23.05.01
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
from torch.utils.data import Dataset, DataLoader

from utils import *
from network import *
from custom_dataset import *
from test_params import parse

args = parse()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.GPU_NUM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_set = test_dataset(args)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 8)

### Network & Data loader setting ###
model = QSMnet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)
load_file_name = args.CHECKPOINT_PATH + 'best_' + args.TAG + '.pth.tar'
checkpoint = torch.load(load_file_name)
model.load_state_dict(checkpoint['state_dict'])
best_epoch = checkpoint['epoch']

loss_total_list = []
NRMSE_total_list = []
PSNR_total_list = []
SSIM_total_list = []
time_total_list = []

if not os.path.exists(args.RESULT_PATH):
    os.makedirs(args.RESULT_PATH)
    
print(f'Best epoch: {best_epoch}\n')

print("------ Testing is started ------")
for i, test_data in enumerate(tqdm(test_loader)):
    subj_name = args.TEST_FILE[i].split('_')[0]
    print(subj_name)
    print(test_data.shape)
    
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
            inferenc_time = time.time() - start_time
            time_list.append(inferenc_time)
            time_total_list.append(inferenc_time)

            l1loss = l1_loss(pred, qsm_batch)

            ### De-normalization ###
            pred_qsm = ((pred[:, 0, ...].cpu() * test_set.qsm_std) + test_set.qsm_mean).to(device).squeeze()
            label_qsm = ((qsm_batch.cpu() * test_set.qsm_std) + test_set.qsm_mean).to(device).squeeze()

            ### Metric calculation ###
            if args.CSF_MASK_EXIST == False:
                nrmse = NRMSE(pred_qsm, label_qsm, m_batch)
                psnr = PSNR(pred_qsm, label_qsm, m_batch)
                ssim = SSIM(pred_qsm.cpu(), label_qsm.cpu(), m_batch.cpu())

            elif args.CSF_MASK_EXIST == True:
                csf_mask_batch = torch.tensor(test_set.csf_mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
                
                nrmse = NRMSE(pred_qsm, label_qsm, csf_mask_batch)
                psnr = PSNR(pred_qsm, label_qsm, csf_mask_batch)
                ssim = SSIM(pred_qsm.cpu(), label_qsm.cpu(), csf_mask_batch.cpu())

            valid_loss_list.append(l1loss.item())
            nrmse_list.append(nrmse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            loss_total_list.append(l1loss.item())
            NRMSE_total_list.append(nrmse)
            PSNR_total_list.append(psnr)
            SSIM_total_list.append(ssim)

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
        
        scipy.io.savemat(args.RESULT_PATH + subj_name + '_' + args.TAG + '.mat',
                         mdict={'label_qsm': test_set.qsm,
                                'label_local': test_set.field,
                                'pred_qsm': pred_qsm_map,
                                'NRMSEmean': NRMSE_mean,
                                'PSNRmean': PSNR_mean,
                                'SSIMmean': SSIM_mean,
                                'NRMSEstd': NRMSE_std,
                                'PSNRstd': PSNR_std,
                                'SSIMstd': SSIM_std,
                                'inference_time': total_time})

total_loss_mean = np.mean(loss_total_list)
total_NRMSE_mean = np.mean(NRMSE_total_list)
total_PSNR_mean = np.mean(PSNR_total_list)
total_SSIM_mean = np.mean(SSIM_total_list)

total_loss_std = np.std(loss_total_list)
total_NRMSE_std = np.std(NRMSE_total_list)
total_PSNR_std = np.std(PSNR_total_list)
total_SSIM_std = np.std(SSIM_total_list)

print(f'NRMSE: {total_NRMSE_mean}, {total_NRMSE_std}  PSNR: {total_PSNR_mean}, {total_PSNR_std}  SSIM: {total_SSIM_mean}, {total_SSIM_std}  Loss: {total_loss_mean}, {total_loss_std}')
print(f'Total inference time: {np.mean(time_total_list)}')

print("------ Testing is finished ------")