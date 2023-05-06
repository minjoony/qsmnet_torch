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
# Last update: 23.05.06
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
import datetime
import matplotlib.pyplot as plt
import logging_helper as logging_helper

from collections import OrderedDict

from utils import *
from network import *
from custom_dataset import *
from test_params import parse

args = parse()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.GPU_NUM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Logger setting ###
logger = logging.getLogger("module.test")
logger.setLevel(logging.INFO)
logging_helper.setup(args.CHECKPOINT_PATH + 'Results', 'test_log.txt')

nowDate = datetime.datetime.now().strftime('%Y-%m-%d')
nowTime = datetime.datetime.now().strftime('%H:%M:%S')
logger.info(f'Date: {nowDate}  {nowTime}')

for key, value in vars(args).items():
    logger.info('{:15s}: {}'.format(key,value))
    

### Data & network setting ###
test_set = test_dataset(args)

model = QSMnet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)
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
    logger.info(f'Multi GPU - num: {torch.cuda.device_count()} - were used')
    model.load_state_dict(new_state_dict)
    
best_epoch = checkpoint['epoch']

loss_total_list = []
NRMSE_total_list = []
PSNR_total_list = []
SSIM_total_list = []
time_total_list = []

if not os.path.exists(args.RESULT_PATH):
    os.makedirs(args.RESULT_PATH)
    
logger.info(f'Best epoch: {best_epoch}\n')

logger.info("------ Testing is started ------")

for idx in range(0, len(args.TEST_FILE)):
    subj_name = args.TEST_FILE[idx].split('_')[0]
    
    with torch.no_grad():
        model.eval()
        
        valid_loss_list = []
        nrmse_list = []
        psnr_list = []
        ssim_list = []

        time_list = []

        input_field = test_set.field[idx]
        input_mask = test_set.mask[idx]
        if args.LABEL_EXIST == True:
            label_qsm = test_set.qsm[idx]
        if args.CSF_MASK_EXIST == True:
            input_csf_mask = test_set.csf_mask[idx]
        matrix_size = test_set.matrix_size[idx]
        
        if len(matrix_size) == 3:
            ### Case of single head-orientation: expanding dim ###
            matrix_size_list = list(matrix_size)
            matrix_size_list.append(1)
            matrix_size = tuple(matrix_size_list)
            
            input_field = np.expand_dims(input_field, 3)
            input_mask = np.expand_dims(input_mask, 3)
            if args.LABEL_EXIST == True:
                label_qsm = np.expand_dims(label_qsm, 3)
            if args.CSF_MASK_EXIST == True:
                input_csf_mask = np.expand_dims(input_csf_mask, 3)

        input_field_map = np.zeros(matrix_size)
        pred_qsm_map = np.zeros(matrix_size)
        label_qsm_map = np.zeros(matrix_size)
        
        for direction in range(matrix_size[-1]):
            ### Setting dataset & normalization & masking ###
            local_f_batch = torch.tensor(input_field[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
            local_f_batch = ((local_f_batch.cpu() - test_set.field_mean) / test_set.field_std).to(device) # normalization

            m_batch = torch.tensor(input_mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
            
            local_f_batch = local_f_batch * m_batch # masking
            
            if args.LABEL_EXIST == True:
                qsm_batch = torch.tensor(label_qsm[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                qsm_batch = ((qsm_batch.cpu() - test_set.qsm_mean) / test_set.qsm_std).to(device) # normalization
                qsm_batch = qsm_batch * m_batch # masking

            ### Inference ###
            # input: local_f_batch (dim: [batch_size, 1, 64, 64, 64])
            # label: qsm_batch (dim: [batch_size, 1, 64, 64, 64])
            start_time = time.time()
            pred_batch = model(local_f_batch)
            inferenc_time = time.time() - start_time
            time_list.append(inferenc_time)
            time_total_list.append(inferenc_time)


            ### De-normalization (input & output) ###
            pred_qsm = ((pred_batch[:, 0, ...].cpu() * test_set.qsm_std) + test_set.qsm_mean).to(device).squeeze() # denormalization
                
            if args.LABEL_EXIST == True:
                label_qsm_for_metric = ((qsm_batch.cpu() * test_set.qsm_std) + test_set.qsm_mean).to(device).squeeze() # denormalization
                
                ### Metric calculation ###
                l1loss = l1_loss(pred_batch, qsm_batch)
                if args.CSF_MASK_EXIST == True:
                    csf_mask_batch = torch.tensor(input_csf_mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()

                    nrmse = NRMSE(pred_qsm, label_qsm_for_metric, csf_mask_batch)
                    psnr = PSNR(pred_qsm, label_qsm_for_metric, csf_mask_batch)
                    ssim = SSIM(pred_qsm.cpu(), label_qsm_for_metric.cpu(), csf_mask_batch.cpu())
                elif args.CSF_MASK_EXIST == False:
                    nrmse = NRMSE(pred_qsm, label_qsm_for_metric, m_batch)
                    psnr = PSNR(pred_qsm, label_qsm_for_metric, m_batch)
                    ssim = SSIM(pred_qsm.cpu(), label_qsm_for_metric.cpu(), m_batch.cpu())

                valid_loss_list.append(l1loss.item())
                nrmse_list.append(nrmse)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                loss_total_list.append(l1loss.item())
                NRMSE_total_list.append(nrmse)
                PSNR_total_list.append(psnr)
                SSIM_total_list.append(ssim)
                
                label_qsm_map[...,  direction] = label_qsm_for_metric.cpu() * m_batch.cpu()

            input_field_map[..., direction] = input_field[..., direction]
            pred_qsm_map[..., direction] = pred_qsm.cpu() * m_batch.cpu()
            
            torch.cuda.empty_cache();

        if args.LABEL_EXIST == True:
            test_loss = np.mean(valid_loss_list)
            NRMSE_mean = np.mean(nrmse_list)
            PSNR_mean = np.mean(psnr_list)
            SSIM_mean = np.mean(ssim_list)

            NRMSE_std = np.std(nrmse_list)
            PSNR_std = np.std(psnr_list)
            SSIM_std = np.std(ssim_list)
            total_time = np.mean(time_list)
            logger.info(f'{subj_name} - NRMSE: {NRMSE_mean:.4f}, {NRMSE_std:.4f}  PSNR: {PSNR_mean:.4f}, {PSNR_std:.4f}  SSIM: {SSIM_mean:.4f}, {SSIM_std:.4f}  Loss: {test_loss:.4f}')

            scipy.io.savemat(args.RESULT_PATH + args.RESULT_FILE + subj_name + '_' + args.TAG + '.mat',
                             mdict={'input_local': input_field_map,
                                    'label_qsm': label_qsm_map,
                                    'pred_qsm': pred_qsm_map,
                                    'NRMSEmean': NRMSE_mean,
                                    'PSNRmean': PSNR_mean,
                                    'SSIMmean': SSIM_mean,
                                    'NRMSEstd': NRMSE_std,
                                    'PSNRstd': PSNR_std,
                                    'SSIMstd': SSIM_std,
                                    'inference_time': total_time})
        else:
            total_time = np.mean(time_list)
            scipy.io.savemat(args.RESULT_PATH + args.RESULT_FILE + subj_name + '_' + args.TAG + '.mat',
                             mdict={'input_local': input_field_map,
                                    'pred_qsm': pred_qsm_map,
                                    'inference_time': total_time})

if args.LABEL_EXIST == True:
    total_loss_mean = np.mean(loss_total_list)
    total_NRMSE_mean = np.mean(NRMSE_total_list)
    total_PSNR_mean = np.mean(PSNR_total_list)
    total_SSIM_mean = np.mean(SSIM_total_list)

    total_loss_std = np.std(loss_total_list)
    total_NRMSE_std = np.std(NRMSE_total_list)
    total_PSNR_std = np.std(PSNR_total_list)
    total_SSIM_std = np.std(SSIM_total_list)

    logger.info(f'\n Total NRMSE: {total_NRMSE_mean:.4f}, {total_NRMSE_std:.4f}  PSNR: {total_PSNR_mean:.4f}, {total_PSNR_std:.4f}  SSIM: {total_SSIM_mean:.4f}, {total_SSIM_std:.4f}  Loss: {total_loss_mean:.4f}, {total_loss_std:.4f}')
logger.info(f'Total inference time: {np.mean(time_total_list)}')

logger.info("------ Testing is finished ------")