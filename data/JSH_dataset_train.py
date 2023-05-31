import random
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as data
import data.util as util
from models.extrac_S import extrac_structure

class JSHDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR), GT and noisy image pairs.
    If only GT and noisy images are provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(JSHDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT, self.paths_Noisy = None, None, None
        self.sizes_LQ, self.sizes_GT, self.sizes_Noisy = None, None, None
        self.LQ_env, self.GT_env, self.Noisy_env = None, None, None  # environment for mat

        with h5py.File(self.opt['dataroot_HDR'], 'r') as f:
            self.length = len(f['HDR_data'])

        self.random_scale_list = [1]

    def __getitem__(self, index):
        if not hasattr(self, 'file_HDR'):
            self.file_HDR = h5py.File(self.opt['dataroot_HDR'], 'r')
            self.file_HDR = self.file_HDR['HDR_data']
        if not hasattr(self, 'file_SDR'):
            self.file_SDR = h5py.File(self.opt['dataroot_SDR'], 'r')
            self.file_SDR = self.file_SDR['SDR_data']
        scale = self.opt['scale']

        # get GT image
        HDR_img = self.file_HDR[index]/1023
        SDR_img = self.file_SDR[index]/255

        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            SDR_img = util.modcrop(SDR_img, scale)
            HDR_img = util.modcrop(HDR_img, scale)

        if self.opt['phase'] == 'train':
            C, H, W = SDR_img.shape

            HDR_img_resize = HDR_img
            # augmentation - flip, rotate
            HDR_img, SDR_img, HDR_img_resize = util.augment([HDR_img, SDR_img,HDR_img_resize], self.opt['use_flip'],
                                          self.opt['use_rot'])

        S = extrac_structure(HDR_img, SDR_img)
        # BGR to RGB, HWC to CHW, numpy to tensor
        HDR_img = torch.from_numpy(np.ascontiguousarray(HDR_img)).float()
        SDR_img = torch.from_numpy(np.ascontiguousarray(SDR_img)).float()
        HDR_img_resize = torch.from_numpy(np.ascontiguousarray(HDR_img_resize)).float()

        return {'SDR_img': SDR_img,  'HDR_img': HDR_img, 'HDR_img_resize': HDR_img_resize, 'S':S}

    def __len__(self):
        return self.length


