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

        with h5py.File(self.opt['dataroot_HDR'], 'r') as f:
            self.length = len(f['HDR'])

        self.random_scale_list = [1]

    def __getitem__(self, index):
        if not hasattr(self, 'file_HDR'):
            self.file_HDR = h5py.File(self.opt['dataroot_HDR'], 'r')
            self.file_HDR = self.file_HDR['HDR']
        if not hasattr(self, 'file_SDR'):
            self.file_SDR = h5py.File(self.opt['dataroot_SDR'], 'r')
            self.file_SDR = self.file_SDR['SDR_YUV']

        # get GT image
        HDR_img = self.file_HDR[index]/1023.0

        # get Noisy image
        SDR_img = self.file_SDR[index]/255.0

        S = extrac_structure(HDR_img, SDR_img)
        HDR_img = torch.from_numpy(np.ascontiguousarray(HDR_img)).float()
        SDR_img = torch.from_numpy(np.ascontiguousarray(SDR_img)).float()

        return {'SDR_img': SDR_img,  'HDR_img': HDR_img, 'S':S}


    def __len__(self):
        return self.length
