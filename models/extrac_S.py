import torch
from models.modules.loss import TV_extractor
import torch.nn as nn
import numpy as np
from models.modules.lap_pyramid import Lap_Pyramid_Bicubic

def norm(img):
    max = np.max(img)
    min = np.min(img)
    return (img-min)/ (max-min)

def IN_EMLF(img):
    tv = TV_extractor()
    # in_norm = nn.InstanceNorm2d(3)
    img_LF = (img)
    img_tv = tv(img_LF)
    return img_tv


def laplacian(img, factor):
    lap_pyramid = Lap_Pyramid_Bicubic(1)
    out = lap_pyramid.pyramid_decom(img/factor)
    HF = out[0]*factor
    LF = out[1]*factor
    return LF, HF

def extract_S(UHDHDR, SDR):

    LF = UHDHDR
    HDR = IN_EMLF(LF)
    SDR = IN_EMLF(SDR)
    HDR = HDR.numpy()
    SDR = SDR.numpy()
    HDR = np.power(HDR, 1.5)
    SDR = np.power(SDR, 1.5)
    out = (SDR + HDR)/2

    return out, HDR, SDR

def extrac_structure(hdr_img, sdr_img):
    # laplacian in 255, out 255
    hd_hdr_img = torch.from_numpy(hdr_img[0,:,:][None,None, ::]/1.0)
    hd_hdr_img, HF = laplacian(hd_hdr_img, 255.0)

    #tv loss in 255
    hdr = np.array(hd_hdr_img)/1.0
    sdr = np.array(sdr_img)[0,:,:][None, None,::]/1.0
    hdr = torch.from_numpy(hdr).float()
    sdr = torch.from_numpy(sdr).float()
    out, HDR, SDR = extract_S(hdr, sdr)

    return out[0]