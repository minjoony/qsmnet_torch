'''
#
# Description:
#  Test parameters for qsmnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 23.05.01
'''
"""
Experiment setting parameters
    GPU_NUM: number of GPU to use.
    INPUT_UNIT: unit of input maps ('radian' or 'Hz')
    TAG: type of trained network (loss: best loss model, nrmse: best nrmse model, psnr: best psnr model)
    LABEL_EXIST: bool value if ground-truth label exist (True: ground-truth exist, False: ground-truth not exist)
    CSF_MASK_EXIST: bool value if CSF mask exist (True: CSF mask exist, False: CSF mask not exist)
    
    TEST_PATH: path of test dataset.
    TEST_FILE: filename of test dataset.
    
    CHECKPOINT_PATH: path of network(x-sepnet) checkpoint.
    CHECKPOINT_FILE: filename of network(x-sepnet) checkpoint.
    
    PRE_NET_CHECKPOINT_PATH: path of pre-network(QSMnet) checkpoint.
    PRE_NET_CHECKPOINT_FILE: filename of pre-network(QSMnet) checkpoint.
    
    VALUE_FILE_PATH: path of file of normalization factors used in training
    VALUE_FILE_NAME: filename of normalization factors used in training
    
    RESULT_PATH: path to save the results.
    RESULT_FILE: filename of result file (mat).
"""
GPU_NUM = '2'
INPUT_UNIT = 'ppm'
TAG = 'nrmse'
LABEL_EXIST = False
CSF_MASK_EXIST = False

TEST_PATH = '../Data/Test/'
TEST_FILE = ['MSsubj2_calcification_DataFor_xsepnet_ppm.mat']
# TEST_FILE = ['subj05_DataFor_xsepnet_ppm_COSMOS_6dir_final.mat', 'subj06_DataFor_xsepnet_ppm_COSMOS_6dir_final.mat', 'subj08_DataFor_xsepnet_ppm_COSMOS_6dir_final', 'subj14_DataFor_xsepnet_ppm_COSMOS_6dir_final.mat']

CHECKPOINT_PATH = './Checkpoint/230428_qsmnet_cosmos_stepLR/'
CHECKPOINT_FILE = 'best_nrmse.pth.tar'

VALUE_FILE_PATH = '../Data/'
VALUE_FILE_NAME = 'xsepnet_train_patch_norm_factor.mat'

RESULT_PATH = CHECKPOINT_PATH + 'Results/'
RESULT_FILE = 'qsmnetplus_'


"""
Physics-parameters
    gyro: gyromagnetic ratio
    delta_TE: time gap between multi-echo times
    CF: center frequency (used for Hz -> ppm calculation)
    Dr: relaxometric constrant between R2' and susceptibility
    
    * Input of network must be unit of ppm. So If the input map has unit of Hz, you can change it to ppm by entering physics-params below (only in case of inference).
    ** Ref: Yoon, Jaeyeon, et al. "Quantitative susceptibility mapping using deep neural network: QSMnet." Neuroimage 179 (2018): 199-206.
"""
delta_TE = 0.005
CF = 123177385
Dr = 114


"""
Network-parameters
    CHANNEL_IN: number of out-channels for first conv layers
    KERNEL_SIZE: kernel size of conv layers
"""
CHANNEL_IN = 32
KERNEL_SIZE = 3


import argparse

def parse():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--GPU_NUM", default=GPU_NUM)
    parser.add_argument("--INPUT_UNIT", default=INPUT_UNIT)
    parser.add_argument("--TAG", default=TAG)
    parser.add_argument("--LABEL_EXIST", default=LABEL_EXIST)
    parser.add_argument("--CSF_MASK_EXIST", default=CSF_MASK_EXIST)

    parser.add_argument("--TEST_PATH", default=TEST_PATH)
    parser.add_argument("--TEST_FILE", default=TEST_FILE)
    
    parser.add_argument("--CHECKPOINT_PATH", default=CHECKPOINT_PATH)
    parser.add_argument("--CHECKPOINT_FILE", default=CHECKPOINT_FILE)
    
    parser.add_argument("--VALUE_FILE_PATH", default=VALUE_FILE_PATH)
    parser.add_argument("--VALUE_FILE_NAME", default=VALUE_FILE_NAME)
    
    parser.add_argument("--RESULT_PATH", default=RESULT_PATH)
    parser.add_argument("--RESULT_FILE", default=RESULT_FILE)

    parser.add_argument("--delta_TE", default=delta_TE)
    parser.add_argument("--CF", default=CF)
    parser.add_argument("--Dr", default=Dr)

    parser.add_argument("--CHANNEL_IN", default=CHANNEL_IN)
    parser.add_argument("--KERNEL_SIZE", default=KERNEL_SIZE)
    
    args = parser.parse_args()
    return args