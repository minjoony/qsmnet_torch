'''
#
# Description:
#  Util codes for qsmnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 23.07.13
'''
import os
import math
import numpy as np
import h5py
import scipy.io
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log10, sqrt
from numpy.fft import fftn, ifftn, fftshift
from skimage.metrics import structural_similarity as ssim        
        
def Concat(x, y):
    return torch.cat((x,y),1)


class Conv3d(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=k_size, stride=1, padding=int(k_size/2), dilation=1)
        self.bn = nn.BatchNorm3d(c_out)
        self.act = nn.ReLU()
        nn.init.xavier_uniform_(self.conv.weight)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    
class Conv(nn.Module):
    def __init__(self, c_in, c_out):
        super(Conv, self).__init__()
        self.conv=nn.Conv3d(c_in,  c_out, kernel_size=1, stride=1, padding=0, dilation=1)
        nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self,x):
        return self.conv(x)
    
    
class Pool3d(nn.Module):
    def __init__(self):
        super(Pool3d, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1)
    
    def forward(self,x):
        return self.pool(x)
    
    
class Deconv3d(nn.Module):
    def __init__(self, c_in, c_out):
        super(Deconv3d, self).__init__()
        self.deconv=nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2, padding=0, dilation=1)
        nn.init.xavier_uniform_(self.deconv.weight)
    
    def forward(self,x):
        return self.deconv(x)

    
def l1_loss(x, y):
    return torch.abs(x-y).mean()


def model_loss(pred_qsm, label_local_f, m, d):
    """
    Args:
        pred_qsm (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. predicted qsm map.
        label_local_f (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. local field map.
        m (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. mask.
        d (ndarray): (s1, s2, s3) size matrix. dipole kernel.
    Description:
        [ local field map = sus map * dipole kernel ] in image domain
        using FFT, we can use multiplication instead of convolution.
    """
    num_batch = pred_qsm.shape[0]
    device = pred_qsm.device;
    
    ### FFT(pred sus map) x dipole kernel ###
    pred_qsm = torch.stack((pred_qsm, torch.zeros(pred_qsm.shape, dtype=pred_qsm.dtype, device=device)), dim=-1)
    fft_p = torch.fft.fft(pred_qsm, 3)
    # dtype = fft_p.dtype;
    
    d = d[np.newaxis, np.newaxis, ...]
    d = torch.tensor(d, dtype=pred_qsm.dtype, device=device).repeat(num_batch, 1, 1, 1, 1)
    d = torch.stack((d, torch.zeros(d.shape, dtype=pred_qsm.dtype, device=device)), dim=-1)
    
    y = torch.zeros(pred_qsm.shape, dtype=pred_qsm.dtype, device=device)
    y[..., 0] = fft_p[..., 0] * d[..., 0] - fft_p[..., 1] * d[..., 1] # real part
    y[..., 1] = fft_p[..., 0] * d[..., 1] + fft_p[..., 1] * d[..., 0] # imaginary part
    
    ### IFT results = pred susceptibility map * dipole kernel ###
    y = torch.fft.ifft(y, 3)
    pred_local_f = y[..., 0]
    
    ############################################################################################################
#     pred_local_f_temp = pred_local_f.cpu().detach().numpy() * m.cpu().detach().numpy()
#     label_local_f_temp = label_local_f.cpu().detach().numpy() * m.cpu().detach().numpy()
        
#     scipy.io.savemat('./predMaps_train_field.mat',
#              mdict={'pred_local': pred_local_f_temp,
#                     'label_local': label_local_f_temp})
    ############################################################################################################
    
    local_f_loss = l1_loss(label_local_f*m, pred_local_f*m)
    
    return local_f_loss


def grad_loss(x, y):
    device = x.device
    x_cen = x[:,:,1:-1,1:-1,1:-1]
    grad_x = torch.zeros(x_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = x[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(x_cen.shape, device=device)
                else:
                    temp = torch.relu(x_slice-x_cen)/s
                grad_x = grad_x + temp
    
    y_cen = y[:,:,1:-1,1:-1,1:-1]
    grad_y = torch.zeros(y_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                y_slice = y[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(y_cen.shape, device=device)
                else:
                    temp = torch.relu(y_slice-y_cen)/s
                grad_y = grad_y + temp
    
    return l1_loss(grad_x, grad_y)


def total_loss(p, y, x, m, d, w_l1, w_md, w_gd, qsm_mean, qsm_std, local_f_mean, local_f_std):
    """
    Args:
        p (torch.tensor): (batch_size, 2, s1, s2, s3) size matrix. predicted susceptability map.
        y (torch.tensor): (batch_size, 2, s1, s2, s3) size matrix. susceptability map (label).
        x (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. local field map.
        m (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. mask.
        d (ndarray): (s1, s2, s3) size matrix. dipole kernel.
        w_l1 (float): weighting factor for L1 losses
        w_md (float): weighting factor for model losses
        w_gd (float): weighting factor for gradient losses

    Returns:
        l1loss (torch.float): L1 loss. 
        mdloss (torch.float): model loss
        gdloss (torch.float): gradient loss
        tloss (torch.float): total loss. sum of above three losses with weighting factor
    """
    ### Splitting into positive/negative maps & masking ###
    pred_qsm = p * m
    label_qsm = y * m
    local_f = x * m
    
    ### L1 loss ###
    l1loss = l1_loss(pred_qsm, label_qsm)
    
    ### Gradient loss ###
    gdloss = grad_loss(pred_qsm, label_qsm)
    
    ### De-normalization ###
    device = p.device
    pred_qsm = torch.from_numpy((pred_qsm.cpu().detach().numpy() * qsm_std) + qsm_mean).to(device)
    local_f = torch.from_numpy((local_f.cpu().detach().numpy() * local_f_std) + local_f_mean).to(device)

    ### Model loss ###
    mdloss = model_loss(pred_qsm, local_f, m, d)
        
    total_loss = l1loss * w_l1 + mdloss * w_md + gdloss * w_gd
    return total_loss, l1loss, mdloss, gdloss


def dipole_kernel(matrix_size, voxel_size, B0_dir):
    """
    Args:
        matrix_size (array_like): should be length of 3.
        voxel_size (array_like): should be length of 3.
        B0_dir (array_like): should be length of 3.
        
    Returns:
        D (ndarray): 3D dipole kernel matrix in Fourier domain.  
    """    
    x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
    y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
    z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
    Y, X, Z = np.meshgrid(x, y, z)
    
    X = X/(matrix_size[0]*voxel_size[0])
    Y = Y/(matrix_size[1]*voxel_size[1])
    Z = Z/(matrix_size[2]*voxel_size[2])
    
    D = 1/3 - (X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2])**2/((X**2 + Y**2 + Z**2) + 1e-6)
    D[np.isnan(D)] = 0
    D = fftshift(D)
    return D


def save_model(epoch, model, PATH, TAG):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict()},
        f'{PATH}/{TAG}.pth.tar')
    torch.save(model, f'{PATH}/model.pt')
    
    
def NRMSE(im1, im2, mask):
    im1 = im1 * mask
    im2 = im2 * mask
    
    mse = torch.mean((im1-im2)**2)
    nrmse = sqrt(mse)/sqrt(torch.mean(im2**2))
    
#     im1 = im1 * mask
#     im2 = im2 * mask
    
#     mse = torch.mean((im1[mask]-im2[mask])**2)
#     nrmse = sqrt(mse)/sqrt(torch.mean(im2[mask]**2))
    return 100*nrmse


def PSNR(im1, im2, mask):
    im1 = im1 * mask
    im2 = im2 * mask
    
    mse = torch.mean((im1-im2)**2)
    if mse == 0:
        return 100
    #PIXEL_MAX = max(im2[mask])
    PIXEL_MAX = 1
    return 20 * log10(PIXEL_MAX / sqrt(mse))


def SSIM(im1, im2, mask):
    im1 = im1.cpu().detach().numpy(); im2 = im2.cpu().detach().numpy(); mask = mask.cpu().detach().numpy()
    im1 = im1 * mask; im2 = im2 * mask;
    mask = mask.astype(bool)

    # im1 = np.pad(im1,((5,5),(5,5),(5,5)),'constant',constant_values=(0))   
    # im2 = np.pad(im2,((5,5),(5,5),(5,5)),'constant',constant_values=(0)) 
    # mask = np.pad(mask,((5,5),(5,5),(5,5)),'constant',constant_values=(0)).astype(bool) 
    
    min_im = np.min([np.min(im1), np.min(im2)])
    im1[mask] = im1[mask] - min_im
    im2[mask] = im2[mask] - min_im
    
    max_im = np.max([np.max(im1), np.max(im2)])
    im1 = 255 * im1 / max_im
    im2 = 255 * im2 / max_im
    
    if len(im1.shape) == 3:
        ssim_value, ssim_map = ssim(im1, im2, data_range=255, full=True)# gaussian_weights=True, K1=0.01, K2=0.03, full=True)
    
        return np.mean(ssim_map[mask]), ssim_map
    elif len(im1.shape) == 5:
        im1 = im1.squeeze()
        im2 = im2.squeeze()
        mask = mask.squeeze()
        
        if len(im1.shape) == 3:
            im1 = np.expand_dims(im1, axis=0)
            im2 = np.expand_dims(im2, axis=0)
            mask = np.expand_dims(mask, axis=0)
        
        ssim_maps = np.zeros(im1.shape)

        for i in range(0, im1.shape[0]):
            _, ssim_maps[i, :, :, :] = ssim(im1[i, :, :, :], im2[i, :, :, :], data_range=255, full=True)# gaussian_weights=True, K1=0.01, K2=0.03, full=True)
        return np.mean(ssim_maps[mask])
    else:
        raise Exception('SSIM - input dimension error')    
        
        
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
        
def crop_img_16x(img):
    """
    input: 3D img [H, W, C]
    output: cropped 3D img with a H, W of 16x
    """
    if img.shape[0] % 16 != 0:
        residual = img.shape[0] % 16
        img = img[int(residual/2):int(-(residual/2)), :, :]
        
    if img.shape[1] % 16 != 0:
        residual = img.shape[1] % 16
        img = img[:, int(residual/2):int(-(residual/2)), :]
        
    return img


# class train_dataset():
#     def __init__(self, args):
#         data_file = h5py.File(path + 'xsepnet_train_patch.hdf5', "r")
#         value_file = scipy.io.loadmat(path + 'xsepnet_train_patch_norm_factor.mat')
        
#         self.field = data_file['pField']
#         self.qsm = data_file['pCosmosSus']
#         self.mask = data_file['pMask']
        
#         self.field_mean = value_file['field_mean']
#         self.field_std = value_file['field_std']
        
#         self.qsm_mean = value_file['cosmos_sus_mean']
#         self.qsm_std = value_file['cosmos_sus_std']
        
#     def __len__(self):
#         return len(self.mask)

#     def __getitem__(self, idx):
#         # dim: [1, 64, 64, 64]
#         local_f_batch = torch.tensor(self.field[idx,...], dtype=torch.float).unsqueeze(0)
#         qsm_batch = torch.tensor(self.qsm[idx,...], dtype=torch.float).unsqueeze(0)
#         m_batch = torch.tensor(self.mask[idx,...], dtype=torch.float).unsqueeze(0)

#         ### Normalization ###
#         local_f_batch = ((local_f_batch - self.field_mean) / self.field_std)
#         qsm_batch = ((qsm_batch - self.qsm_mean) / self.qsm_std)

#         return idx, local_f_batch, qsm_batch, m_batch


# class valid_dataset():
#     def __init__(self, args):
#         data_file = scipy.io.loadmat(args.VALID_PATH + 'subj10_DataFor_xsepnet_ppm_COSMOS_6dir_final.mat')
#         value_file = scipy.io.loadmat(args.VALUE_PATH + 'xsepnet_train_patch_norm_factor.mat')

#         qsm = data_file['cosmos_4d']
        
#         if args.INPUT_UNIT == 'Hz':
#             ### Converting Hz maps to ppm ###
#             print('Input map unit has been changed (hz -> ppm)')
#             field = data_file['local_f_hz_4d']

#             gyro = args.gyro
#             delta_TE = args.delta_TE
#             CF = args.CF
#             Dr = args.Dr

#             field_in_ppm = -1 * field / (2*math.pi*delta_TE) / CF * 1e6
#         elif args.INPUT_UNIT == 'radian':
#             print('Input map unit has been changed (radian -> ppm)')
#             field = data_file['local_f_4d']
            
#             delta_TE = args.delta_TE
#             CF = args.CF
            
#             field_in_ppm = -1 * field / (2*math.pi*delta_TE) / CF * 1e6
#         elif args.INPUT_UNIT == 'ppm':
#             field = data_file['local_f_ppm_4d']

#             field_in_ppm = field
        
#         self.field = field_in_ppm
#         self.mask = data_file['mask_4d']
#         self.qsm = qsm
        
#         self.field_mean = value_file['field_mean']
#         self.field_std = value_file['field_std']
        
#         self.qsm_mean = value_file['qsm_mean']
#         self.qsm_std = value_file['qsm_std']
        
#         self.matrix_size = self.mask.shape

# def SSIM(im1, im2, mask):
#     im1 = np.copy(im1); im2 = np.copy(im2); mask = mask.cpu().detach().numpy()
#     im1 = im1 * mask; im2 = im2 * mask;
#     mask = mask.astype(bool)

#     min_im = np.min([np.min(im1),np.min(im2)])
#     im1[mask] = im1[mask] - min_im
#     im2[mask] = im2[mask] - min_im
    
#     max_im = np.max([np.max(im1),np.max(im2)])
#     im1 = 255*im1/max_im
#     im2 = 255*im2/max_im

#     _, ssim_map =ssim(im1, im2, data_range=255, gaussian_weights=True, K1=0.01, K2=0.03, full=True)
#     return np.mean(ssim_map[mask])