'''
#
# Description:
#  Test code of x-sepnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 23.04.27
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

from utils import *
from network import *
from custom_dataset import *
from test_params import parse

args = parse()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.GPU_NUM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_set = test_dataset(args)

### Network & Data loader setting ###
pre_model = QSMnet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)
PRE_NET_WEIGHT_NAME = args.PRE_NET_CHECKPOINT_PATH + args.PRE_NET_CHECKPOINT_FILE
PRE_NET_WEIGHT = torch.load(PRE_NET_WEIGHT_NAME)


pre_model.load_state_dict(PRE_NET_WEIGHT['state_dict'])

model = KAInet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)
load_file_name = args.CHECKPOINT_PATH + 'best_' + args.TAG + '.pth.tar'
checkpoint = torch.load(load_file_name)

multi_gpu_used = 0
new_state_dict = OrderedDict()
for name in checkpoint['state_dict']:    
    if name[:6] != 'module':
        break
    else:
        multi_gpu_used = 1
        new_name = name[7:]
        new_state_dict[new_name] = checkpoint['state_dict'][name]

if multi_gpu_used == 0:
    model.load_state_dict(checkpoint['state_dict'])
elif multi_gpu_used == 1:    
    logger.info(f'x-sepnet: multi GPU - num: {torch.cuda.device_count()} - were used')
    model.load_state_dict(new_state_dict)
    
model.load_state_dict(checkpoint['state_dict'])
best_epoch = checkpoint['epoch']

if not os.path.exists(args.RESULT_PATH):
    os.makedirs(args.RESULT_PATH)

print(f'Best epoch: {best_epoch}\n')

print("------ Testing is started ------")
with torch.no_grad():
    pre_model.eval()
    model.eval()
    
    pred_x_pos_map = np.zeros(test_set.matrix_size)
    pred_x_neg_map = np.zeros(test_set.matrix_size)
    pred_sus_map = np.zeros(test_set.matrix_size)

    valid_loss_list = []
    pos_nrmse_list = []
    pos_psnr_list = []
    pos_ssim_list = []

    neg_nrmse_list = []
    neg_psnr_list = []
    neg_ssim_list = []
    
    time_list = []

    for direction in range(test_set.matrix_size[-1]):
        ### Setting dataset & normalization ###
        local_f_batch = torch.tensor(test_set.field[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
        m_batch = torch.tensor(test_set.mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
        
        if args.LABEL_EXIST is True:
            x_pos_batch = torch.tensor(test_set.x_pos[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
            x_neg_batch = torch.tensor(test_set.x_neg[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
            x_pos_batch = ((x_pos_batch.cpu() - test_set.x_pos_mean) / test_set.x_pos_std).to(device)
            x_neg_batch = ((x_neg_batch.cpu() - test_set.x_neg_mean) / test_set.x_neg_std).to(device)
            label_batch = torch.cat((x_pos_batch, x_neg_batch), 1)
            
            if args.CSF_MASK_EXIST is True:
                csf_mask_batch = torch.tensor(test_set.csf_mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()

        if args.INPUT_MAP == 'r2p':
            r2input_batch = torch.tensor(test_set.r2prime[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
            r2input_batch = ((r2input_batch.cpu() - test_set.r2prime_mean) / test_set.r2prime_std).to(device)
            
        elif args.INPUT_MAP == 'r2s':
            r2input_batch = torch.tensor(test_set.r2star[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
            r2input_batch = ((r2input_batch.cpu() - test_set.r2star_mean) / test_set.r2star_std).to(device)
            
        local_f_batch = ((local_f_batch.cpu() - test_set.field_mean) / test_set.field_std).to(device)


        ### Brain masking ###
        local_f_batch = local_f_batch * m_batch
        r2input_batch = r2input_batch * m_batch
        

        ### Extract cosmos batch from pre-QSM Network ###
        pred_cosmos = pre_model(local_f_batch)
        pred_cosmos_batch = pred_cosmos * m_batch
        
        input_batch = torch.cat((pred_cosmos_batch, local_f_batch, r2input_batch), 1)

        start_time = time.time()
        pred = model(input_batch)
        time_list.append(time.time() - start_time)
        
        
        ### De-normalization ###
        pred_cosmos = ((pred_cosmos.cpu() * test_set.cosmos_std) + test_set.cosmos_mean).to(device).squeeze()
        pred_x_pos = ((pred[:, 0, ...].cpu() * test_set.x_pos_std) + test_set.x_pos_mean).to(device).squeeze()
        pred_x_neg = ((pred[:, 1, ...].cpu() * test_set.x_neg_std) + test_set.x_neg_mean).to(device).squeeze()

        
        if args.LABEL_EXIST is True:
            ### De-normalization ###
            label_x_pos = ((x_pos_batch.cpu() * test_set.x_pos_std) + test_set.x_pos_mean).to(device).squeeze()
            label_x_neg = ((x_neg_batch.cpu() * test_set.x_neg_std) + test_set.x_neg_mean).to(device).squeeze()
            
            ### Metric calculation ###
            l1loss = l1_loss(pred, label_batch)
            
            if args.CSF_MASK_EXIST is True:
                pos_nrmse = NRMSE(pred_x_pos, label_x_pos, csf_mask_batch)
                pos_psnr = PSNR(pred_x_pos, label_x_pos, csf_mask_batch)
                pos_ssim = SSIM(pred_x_pos.cpu(), label_x_pos.cpu(), csf_mask_batch.cpu())

                neg_nrmse = NRMSE(pred_x_neg, label_x_neg, csf_mask_batch)
                neg_psnr = PSNR(pred_x_neg, label_x_neg, csf_mask_batch)
                neg_ssim = SSIM(pred_x_neg.cpu(), label_x_neg.cpu(), csf_mask_batch.cpu())
            elif args.CSF_MASK_EXIST is False:
                pos_nrmse = NRMSE(pred_x_pos, label_x_pos, m_batch)
                pos_psnr = PSNR(pred_x_pos, label_x_pos, m_batch)
                pos_ssim = SSIM(pred_x_pos.cpu(), label_x_pos.cpu(), m_batch.cpu())

                neg_nrmse = NRMSE(pred_x_neg, label_x_neg, m_batch)
                neg_psnr = PSNR(pred_x_neg, label_x_neg, m_batch)
                neg_ssim = SSIM(pred_x_neg.cpu(), label_x_neg.cpu(), m_batch.cpu())

            valid_loss_list.append(l1loss.item())
            pos_nrmse_list.append(pos_nrmse)
            pos_psnr_list.append(pos_psnr)
            pos_ssim_list.append(pos_ssim)
            neg_nrmse_list.append(neg_nrmse)
            neg_psnr_list.append(neg_psnr)
            neg_ssim_list.append(neg_ssim)

            pred_sus_map[..., direction] = pred_cosmos_batch.cpu()
            pred_x_pos_map[..., direction] = (pred_x_pos.cpu() * m_batch.cpu())
            pred_x_neg_map[..., direction] = (pred_x_neg.cpu() * m_batch.cpu())

            del(local_f_batch, r2input_batch, x_pos_batch, x_neg_batch, m_batch, input_batch, label_batch, l1loss); torch.cuda.empty_cache();
            
        elif args.LABEL_EXIST is False:
            pred_sus_map[..., direction] = pred_cosmos_batch.cpu()
            pred_x_pos_map[..., direction] = (pred_x_pos.cpu() * m_batch.cpu())
            pred_x_neg_map[..., direction] = (pred_x_neg.cpu() * m_batch.cpu())
            
            del(local_f_batch, r2input_batch, m_batch, input_batch); torch.cuda.empty_cache();
    
    total_time = np.mean(time_list)

    if args.LABEL_EXIST is True:
        test_loss = np.mean(valid_loss_list)
        pos_NRMSE_mean = np.mean(pos_nrmse_list)
        pos_PSNR_mean = np.mean(pos_psnr_list)
        pos_SSIM_mean = np.mean(pos_ssim_list)
        neg_NRMSE_mean = np.mean(neg_nrmse_list)
        neg_PSNR_mean = np.mean(neg_psnr_list)
        neg_SSIM_mean = np.mean(neg_ssim_list)

        pos_NRMSE_std = np.std(pos_nrmse_list)
        pos_PSNR_std = np.std(pos_psnr_list)
        pos_SSIM_std = np.std(pos_ssim_list)
        neg_NRMSE_std = np.std(neg_nrmse_list)
        neg_PSNR_std = np.std(neg_psnr_list)
        neg_SSIM_std = np.std(neg_ssim_list)

        scipy.io.savemat(args.RESULT_PATH + args.RESULT_FILE + '_best_' + args.TAG + '.mat',
                         mdict={'label_x_pos': test_set.x_pos,
                                'label_x_neg': test_set.x_neg,
                                'label_local': test_set.field,
                                'pred_x_pos': pred_x_pos_map,
                                'pred_x_neg': pred_x_neg_map,
                                'pred_sus': pred_sus_map,
                                'posNRMSEmean': pos_NRMSE_mean,
                                'posPSNRmean': pos_PSNR_mean,
                                'posSSIMmean': pos_SSIM_mean,
                                'negNRMSEmean': neg_NRMSE_mean,
                                'negPSNRmean': neg_PSNR_mean,
                                'negSSIMmean': neg_SSIM_mean,
                                'posNRMSEstd': pos_NRMSE_std,
                                'posPSNRstd': pos_PSNR_std,
                                'posSSIMstd': pos_SSIM_std,
                                'negNRMSEstd': neg_NRMSE_std,
                                'negPSNRstd': neg_PSNR_std,
                                'negSSIMstd': neg_SSIM_std})

        print(f'Xpos - NRMSE: {pos_NRMSE_mean}, {pos_NRMSE_std}  PSNR: {pos_PSNR_mean}, {pos_PSNR_std}  SSIM: {pos_SSIM_mean}, {pos_SSIM_std}')
        print(f'Xneg - NRMSE: {neg_NRMSE_mean}, {neg_NRMSE_std}  PSNR: {neg_PSNR_mean}, {neg_PSNR_std}  SSIM: {neg_SSIM_mean}, {neg_SSIM_std}')
    
    elif args.LABEL_EXIST is False:
        scipy.io.savemat(args.RESULT_PATH + args.RESULT_FILE + '_best_' + args.TAG + '.mat',
                         mdict={'pred_sus': pred_sus_map,
                                'pred_x_pos': pred_x_pos_map,
                                'pred_x_neg': pred_x_neg_map})
        
print(f'Total inference time: {total_time}')
print("------ Testing is finished ------")