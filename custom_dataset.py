'''
#
# Description:
#  Dataset codes for qsmnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 23.07.27
'''
import math
import h5py
import scipy.io
import scipy.ndimage
import torch
import mat73
import numpy as np
from utils import *

class train_dataset():
    def __init__(self, args):
        data_file = h5py.File(args.TRAIN_PATH + args.TRAIN_FILE, "r")
        value_file = scipy.io.loadmat(args.VALUE_PATH + args.VALUE_FILE)
        
        self.field = data_file['pField']
        self.qsm = data_file['pCosmosSusbC']
        self.mask = data_file['pMask']
        
        self.field_mean = value_file['field_mean'].item()
        self.field_std = value_file['field_std'].item()
        
        self.qsm_mean = value_file['cosmos_sus_bC_mean'].item()
        self.qsm_std = value_file['cosmos_sus_bC_std'].item()
        
    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        # dim: [1, 64, 64, 64]
        local_f_batch = torch.tensor(self.field[idx,...], dtype=torch.float).unsqueeze(0)
        qsm_batch = torch.tensor(self.qsm[idx,...], dtype=torch.float).unsqueeze(0)
        m_batch = torch.tensor(self.mask[idx,...], dtype=torch.float).unsqueeze(0)

        ### Normalization ###
        local_f_batch = ((local_f_batch - self.field_mean) / self.field_std)
        qsm_batch = ((qsm_batch - self.qsm_mean) / self.qsm_std)

        return idx, local_f_batch, qsm_batch, m_batch


class valid_dataset():
    def __init__(self, args):
        try:
            data_file = scipy.io.loadmat(args.VALID_PATH + args.VALID_FILE)
        except:
            data_file = mat73.loadmat(args.VALID_PATH + args.VALID_FILE)

        value_file = scipy.io.loadmat(args.VALUE_PATH + args.VALUE_FILE)
            
        qsm = data_file['cosmos_bC_4d']
        
        if args.INPUT_UNIT == 'Hz':
            ### Converting Hz maps to ppm ###
            print('Input map unit has been changed (hz -> ppm)')
            field = data_file['local_f_hz_4d']

            CF = args.CF

            field_in_ppm = field / CF * 1e6
        elif args.INPUT_UNIT == 'radian':
            print('Input map unit has been changed (radian -> ppm)')
            field = data_file['local_f_4d']
            
            delta_TE = args.delta_TE
            CF = args.CF
            
            field_in_ppm = -1 * field / (2*math.pi*delta_TE) / CF * 1e6
        elif args.INPUT_UNIT == 'ppm':
            field = data_file['local_f_ppm_4d']

            field_in_ppm = field
        
        
        self.field = field_in_ppm
        self.mask = data_file['mask_4d']
        self.qsm = qsm
        
        self.field_mean = value_file['field_mean'].item()
        self.field_std = value_file['field_std'].item()
        
        self.qsm_mean = value_file['cosmos_sus_bC_mean'].item()
        self.qsm_std = value_file['cosmos_sus_bC_std'].item()
        
        self.matrix_size = self.mask.shape

        
class test_dataset():
    def __init__(self, args):
        value_file = scipy.io.loadmat(args.VALUE_FILE_PATH + args.VALUE_FILE_NAME)

        self.field = []
        self.mask = []
        self.qsm = []
        self.csf_mask = []
        self.matrix_size = []
        
        self.field_mean = value_file['field_mean'].item()
        self.field_std = value_file['field_std'].item()

        self.qsm_mean = value_file['cosmos_sus_bC_mean'].item()
        self.qsm_std = value_file['cosmos_sus_bC_std'].item()
            
        for i in range(0, len(args.TEST_FILE)):
            try:
                data_file = scipy.io.loadmat(args.TEST_PATH + args.TEST_FILE[i])
            except:
                data_file = mat73.loadmat(args.TEST_PATH + args.TEST_FILE[i])


            if args.INPUT_UNIT == 'Hz':
                ### Converting Hz to ppm ###
                print('Input map unit has been changed (hz -> ppm)')
                field = data_file['local_f_hz_4d']

                CF = args.CF

                field_in_ppm = field / CF * 1e6
            elif args.INPUT_UNIT == 'radian':
                ### Converting radian to ppm ###
                print('Input map unit has been changed (radian -> ppm)')
                field = data_file['local_f_4d']

                delta_TE = args.delta_TE
                CF = args.CF

                field_in_ppm = -1 * field / (2*math.pi*delta_TE) / CF * 1e6
            elif args.INPUT_UNIT == 'ppm':
                field = data_file['local_f_ppm_4d']

                field_in_ppm = field

            self.field.append(crop_img_16x(field_in_ppm))
            self.mask.append(crop_img_16x(data_file['mask_4d']))

            if args.LABEL_EXIST is True:
                self.qsm.append(crop_img_16x(data_file['cosmos_bC_4d']))

            if args.CSF_MASK_EXIST is True:
                subj_name = args.TEST_FILE[i].split('_')[0]
                csf_mask_file = scipy.io.loadmat(args.TEST_PATH + subj_name + '_csf_mask_for_metric.mat')

                csf_mask_only = crop_img_16x(csf_mask_file['CSF_mask_4d'])
                csf_mask_only = (csf_mask_only == 0)
                mask_wo_csf = self.mask[i] * csf_mask_only
                
                ### Vessel masking-out ###
                # vessel_mask_file = scipy.io.loadmat(args.TEST_PATH + subj_name + '_csf_mask_for_metric_vessel.mat')
                # vessel_mask_only = crop_img_16x(vessel_mask_file['vessel_mask_4d'])
                # vessel_mask_only = (vessel_mask_only == 0)
                # mask_wo_vessel = mask_wo_csf * vessel_mask_only
                
                self.csf_mask.append(mask_wo_csf)
                # scipy.io.savemat(args.RESULT_PATH + 'csfmask_.mat', mdict={'mask_vessel': vessel_mask_only,
                                                                           # 'mask_csf': csf_mask_only,
                                                                           # 'mask_wo_vessel': mask_wo_vessel})
            
            self.matrix_size.append(crop_img_16x(data_file['mask_4d']).shape)