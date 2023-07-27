'''
#
# Description:
#  Training parameters for qsmnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 23.07.27
'''
"""
Experiment setting parameters
    GPU_NUM: number of GPU to use.
    SEED: fixed seed for reproducibility.
    INPUT_UNIT: unit of input maps ('Hz' or 'ppm' or 'radian')
    
    TRAIN_PATH: path of train dataset.
    VALID_PATH: path of valid dataset.
    VALUE_PATH: path of value file including mean, std of dataset.
    TRAIN_FILE: filename of train dataset.
    VALID_FILE: filename of valid dataset.
    VALUE_FILE: filename of value file including mean, std of dataset.
    
    CHECKPOINT_PATH = path of network(x-sepnet) checkpoint.
    PRE_NET_CHECKPOINT_PATH: path of pre-network(QSMnet) checkpoint.
    PRE_NET_CHECKPOINT_FILE: filename of pre-network(QSMnet) checkpoint.
"""
GPU_NUM = '5'
SEED = 777
INPUT_UNIT = 'ppm'

TRAIN_PATH = '/fast_storage/minjoon/x-sep/Train/'
VALID_PATH = '../Data/Valid/'
VALUE_PATH = '../Data/'
TRAIN_FILE = 'xsepnet_train_patch_augmented_final.hdf5'
VALID_FILE = 'subj06_DataFor_xsepnet_ppm_COSMOS_6dir_final.mat'
VALUE_FILE = 'xsepnet_train_patch_norm_factor_augmented_final.mat'
CHECKPOINT_PATH = './Checkpoint/230720_qsmnetplus_stepdecay_subj06/'


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
    KERNEL_SIZE: kernel size of conv layers"""
CHANNEL_IN = 32
KERNEL_SIZE = 3

"""
Hyper-parameters
    TRAIN_EPOCH: number of total epoch for training [QSMnet: 50, QSMnet+: 25]
    SAVE_STEP: step for saving the epoch during training
    LEARNING_RATE: learning rate [QSMnet: 0.001, QSMnet+: 0.001]
    LR_EXP_DECAY_GAMMA: multiplicative factor(gamma) of exponential learning rate decay. [QSMnet: 0.9829, QSMnet+: 0.9999]
    BATCH_SIZE: batch size [QSMnet: 12, QSMnet+: 12]
    W_L1Loss: weight of L1 loss [QSMnet: 1, QSMnet+: 1]
    W_MDLOSS: weight of model loss [QSMnet: 1, QSMnet+: 0.5]
    W_GDLOSS: weight of gradient loss [QSMnet: 0.1, QSMnet+: 0.1]
"""
TRAIN_EPOCH = 25
SAVE_STEP = 5
LEARNING_RATE = 0.001
LR_EXP_DECAY_GAMMA = 0.9999
BATCH_SIZE = 12
W_L1Loss = 1
W_MDLOSS = 0.5
W_GDLOSS = 0.1

import argparse

def parse():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--GPU_NUM", default=GPU_NUM)
    parser.add_argument("--SEED", default=SEED)
    parser.add_argument("--INPUT_UNIT", default=INPUT_UNIT)

    parser.add_argument("--TRAIN_PATH", default=TRAIN_PATH)
    parser.add_argument("--VALID_PATH", default=VALID_PATH)
    parser.add_argument("--VALUE_PATH", default=VALUE_PATH)
    parser.add_argument("--TRAIN_FILE", default=TRAIN_FILE)
    parser.add_argument("--VALID_FILE", default=VALID_FILE)
    parser.add_argument("--VALUE_FILE", default=VALUE_FILE)
    parser.add_argument("--CHECKPOINT_PATH", default=CHECKPOINT_PATH)

    parser.add_argument("--delta_TE", default=delta_TE)
    parser.add_argument("--CF", default=CF)
    parser.add_argument("--Dr", default=Dr)


    parser.add_argument("--CHANNEL_IN", default=CHANNEL_IN)
    parser.add_argument("--KERNEL_SIZE", default=KERNEL_SIZE)
    
    parser.add_argument("--TRAIN_EPOCH", default=TRAIN_EPOCH)
    parser.add_argument("--LEARNING_RATE", default=LEARNING_RATE)
    parser.add_argument("--LR_EXP_DECAY_GAMMA", default=LR_EXP_DECAY_GAMMA)
    parser.add_argument("--BATCH_SIZE", default=BATCH_SIZE)
    parser.add_argument("--SAVE_STEP", default=SAVE_STEP)
    parser.add_argument("--W_L1LOSS", default=W_L1Loss)
    parser.add_argument("--W_MDLOSS", default=W_MDLOSS)
    parser.add_argument("--W_GDLOSS", default=W_GDLOSS)

    args = parser.parse_args()
    return args