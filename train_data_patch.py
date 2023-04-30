import scipy.io
import numpy as np
import h5py
import time
import os
import sys
import math
import shutil

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
File Path
'''
FILE_PATH_INPUT = './data/'
FILE_NAME = '/DataFor_xsep_COSMOS_newDr.mat'
FILE_PATH_OUTPUT = './'
result_file = h5py.File(FILE_PATH_OUTPUT + 'train_patch.hdf5', 'w')

'''
Constant Variables
    PS: Patch size
    dir_num: number of directions
    patch_num: Order of Dimension: [x, y, z]
    
    gyro, delta_TE, CF, Dr: Variables to convert Hz to ppm
'''
PS = 64
patch_num = [6, 8, 7]

gyro = 42.5775e6
delta_TE = 0.005
CF = 123177385
Dr = 114


if os.path.isdir(FILE_PATH_INPUT + "/.ipynb_checkpoints"):
    shutil.rmtree(FILE_PATH_INPUT + "/.ipynb_checkpoints")
    
patches_cosmos_sus = []
patches_field = []
patches_mask = []
patches_r2star = []
patches_x_pos = []
patches_x_neg = []

subject_list = os.listdir(FILE_PATH_INPUT)
print("Subjects num:", len(subject_list))
print("Subjects:", subject_list)

print("*** Patching start !!! ***")
start_time = time.time()

m = scipy.io.loadmat(FILE_PATH_INPUT + FILE_NAME)

cosmos_sus = m['cosmos_4d']
field = m['local_f_4d']
mask = m['mask_4d']


### Crop brain region tightly ###
y_max = np.max(np.where(mask[:, :, :, 0] != 0)[0])
x_max = np.max(np.where(mask[:, :, :, 0] != 0)[1])
z_max = np.max(np.where(mask[:, :, :, 0] != 0)[2])

y_min = np.min(np.where(mask[:, :, :, 0] != 0)[0])
x_min = np.min(np.where(mask[:, :, :, 0] != 0)[1])
z_min = np.min(np.where(mask[:, :, :, 0] != 0)[2])

new_y_min = int(y_min-PS/2)
new_y_max = int(y_max+PS/2)

new_x_min = int(x_min-PS/2)
new_x_max = int(x_max+PS/2)

new_z_min = int(z_min-PS/2)
new_z_max = int(z_max+PS/2)

print('y:', new_y_min, new_y_max)
print('x:', new_x_min, new_x_max)
print('z:', new_z_min, new_z_max)

if(new_y_min < 0):
    new_y_min = 0

if(new_z_min < 0):
    new_z_min = 0

if(new_x_min < 0):
    new_x_min = 0

if(new_y_max > mask.shape[0]):
    new_y_max = mask.shape[0]-1

if(new_x_max > mask.shape[1]):
    new_x_max = mask.shape[1]-1

if(new_z_max > mask.shape[2]):
    new_z_max = mask.shape[2]-1

print('y:', new_y_min, new_y_max)
print('x:', new_x_min, new_x_max)
print('z:', new_z_min, new_z_max)

origin_mask = mask.copy()
print('Origin size:', np.shape(origin_mask))

cosmos_sus = cosmos_sus[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
field = field[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
mask = mask[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]

#     scipy.io.savemat(FILE_PATH_OUTPUT + 'Result/tightMap/' + subject + '_tight_mask.mat',
#                      mdict={'origin_mask':origin_mask, 'tight_mask': mask})

### Converting Hz maps to ppm ###
field_in_ppm = -1 * field / (2*math.pi*delta_TE) / CF * 1e6

matrix_size = np.shape(mask)
strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)];

# print('Subject name:', subject)
print('Matrix size:', matrix_size)
print('Strides:', strides)

for direction in range(matrix_size[-1]):
    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            for k in range(patch_num[2]):
                patches_cosmos_sus.append(cosmos_sus[
                               i * strides[0]:i * strides[0] + PS,
                               j * strides[1]:j * strides[1] + PS,
                               k * strides[2]:k * strides[2] + PS,
                               direction])
                patches_field.append(field_in_ppm[
                               i * strides[0]:i * strides[0] + PS,
                               j * strides[1]:j * strides[1] + PS,
                               k * strides[2]:k * strides[2] + PS,
                               direction])
                patches_mask.append(mask[
                               i * strides[0]:i * strides[0] + PS,
                               j * strides[1]:j * strides[1] + PS,
                               k * strides[2]:k * strides[2] + PS,
                               direction])
                

print("*** Patching Done !!! ***")
print("*** Saving start !!! ***")

patches_cosmos_sus = np.array(patches_cosmos_sus, dtype='float32', copy=False)
patches_field = np.array(patches_field, dtype='float32', copy=False)
patches_mask = np.array(patches_mask, dtype='float32', copy=False)

cosmos_sus_mean = np.mean(patches_cosmos_sus[patches_mask > 0])
cosmos_sus_std = np.std(patches_cosmos_sus[patches_mask > 0])
field_mean = np.mean(patches_field[patches_mask > 0])
field_std = np.std(patches_field[patches_mask > 0])
n_element = np.sum(patches_mask)

result_file.create_dataset('pCosmosSus', data=patches_cosmos_sus)
result_file.create_dataset('pField', data=patches_field)
result_file.create_dataset('pMask', data=patches_mask)

scipy.io.savemat(FILE_PATH_OUTPUT + 'train_patch_norm_factor.mat',
                 mdict={'cosmos_sus_mean': cosmos_sus_mean, 'cosmos_sus_std': cosmos_sus_std,
                        'field_mean': field_mean, 'field_std': field_std,
                        'n_element': n_element})
print("Final input data size : " + str(np.shape(patches_mask)))

del patches_cosmos_sus
del patches_field
del patches_mask
result_file.close()
    
print("*** Saving Done !!! ***")