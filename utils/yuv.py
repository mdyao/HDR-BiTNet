import numpy as np
import os
import sys
import cv2
import numpy as np
import struct
import sys
import glob
import h5py
from datetime import datetime

def compute_psnr(img_orig, img_out, peak):
    mse = np.mean(np.square(img_orig - img_out))
    psnr = 10 * np.log10(peak * peak / mse)
    return psnr

# %%% Save YUV frame of SDR or HDR videos %%%
# % 'SDR_HDR' flag should be specified with either 'SDR' or 'HDR'
def save_yuv(data, new_file, height, width, h_factor, w_factor, SDR_HDR):
    # % get size of data
    datasize = data.shape
    datasizelength = len(datasize)

    # % subsampling of U and V
    if datasizelength == 2 or h_factor == 0:
        # %4:0:0
        y = np.zeros((height, width))
        y[1:height, 1:width] = data[:, :, 1]
    elif datasizelength == 3:
        y = np.zeros((height, width))
        u = np.zeros((height, width))
        v = np.zeros((height, width))

        y[0: height, 0: width] = data[:, :, 0] * 1.0
        u[0: height, 0: width] = data[:, :, 1] * 1.0
        v[0: height, 0: width] = data[:, :, 2] * 1.0

        if w_factor == 1:
            # % 4: 1:1
            u2 = u
            v2 = v
        elif h_factor == 0.5:
            # % 4: 2:0
            u2 = np.zeros((height // 2, width // 2))
            u2[0: height // 2, 0: width // 2] = u[np.round(range(0, u.shape[0], 2)).astype(int)[:, None],
                                                  np.round(range(0, u.shape[1], 2)).astype(int)] + \
                                                u[np.round(range(0, u.shape[0], 2)).astype(int)[:, None],
                                                  np.round(range(1, u.shape[1], 2)).astype(int)] + \
                                                u[np.round(range(1, u.shape[0], 2)).astype(int)[:, None],
                                                  np.round(range(0, u.shape[1], 2)).astype(int)] + \
                                                u[np.round(range(1, u.shape[0], 2)).astype(int)[:, None],
                                                  np.round(range(1, u.shape[1], 2)).astype(int)]
            u2 = u2 / 4
            v2 = np.zeros((height // 2, width // 2))
            v2[0: height // 2, 0: width // 2] = v[np.round(range(0, v.shape[0], 2)).astype(int)[:, None],
                                                  np.round(range(0, v.shape[1], 2)).astype(int)] + \
                                                v[np.round(range(0, v.shape[0], 2)).astype(int)[:, None],
                                                  np.round(range(1, v.shape[1], 2)).astype(int)] + \
                                                v[np.round(range(1, v.shape[0], 2)).astype(int)[:, None],
                                                  np.round(range(0, v.shape[1], 2)).astype(int)] + \
                                                v[np.round(range(1, v.shape[0], 2)).astype(int)[:, None],
                                                  np.round(range(1, v.shape[1], 2)).astype(int)]
            v2 = v2 / 4

    if SDR_HDR == 'HDR':
        y = y.astype(np.uint16)
        u2 = u2.astype(np.uint16)
        v2 = v2.astype(np.uint16)
        # % open file
    elif SDR_HDR == 'SDR':
        y = y.astype(np.uint8)
        u2 = u2.astype(np.uint8)
        v2 = v2.astype(np.uint8)
    with open(new_file, 'ab+') as f:
        if h_factor != 0:
            f.write(np.ascontiguousarray(y))
            f.write(np.ascontiguousarray(u2))
            f.write(np.ascontiguousarray(v2))


# % get factor for YUV-subsampling
def yuv_factor(format):
    fwidth = 0.5
    fheight = 0.5
    if format =='400':
        fwidth = 0
        fheight = 0
    elif format=='411':
        fwidth = 0.25
        fheight = 1
    elif format=='420':
        fwidth = 0.5
        fheight = 0.5
    elif format=='422':
        fwidth = 0.5
        fheight = 1
    elif  format=='444':
        fwidth = 1
        fheight = 1
    return fwidth, fheight

def png2yuv(images, outyuv, HDR_SDR, h, w):
    fwidth, fheight = yuv_factor(format)
    for i in range(0, len(images)):
        print('Processing png2yuv [%02d/%02d]'%((i),len(images)))
        save_yuv(images[i % len(images)], outyuv, h, w, fheight, fwidth, HDR_SDR)

def save_yuv_mp4(images):
    scale = 1
    HDR_SDR = 'HDR'
    fps = 20
    h = 2160//scale
    w = 3840//scale

    # savepath ='/gdata2/yaomd/2022/ideas/ECCV_iHDR/writing/4_decomposition/ifHDR_decom30/sdr.mp4'
    # outyuv = '/gdata2/yaomd/2022/ideas/ECCV_iHDR/writing/4_decomposition/ifHDR_decom30/sdr.yuv'
    savepath ='/data/workspace/2022/ideas/ECCV22/rebuttal/tayler/real_video_hdr.mp4'
    outyuv = '/data/workspace/2022/ideas/ECCV22/rebuttal/tayler/real_video_hdr.yuv'

    # png2yuv(images, outyuv, HDR_SDR, h, w)

    if HDR_SDR=='SDR':
        # cmd = 'ffmpeg -f rawvideo -vcodec rawvideo -s {}x{} -r {} -pix_fmt yuv420p -i {} -vf scale=out_color_matrix=bt709:out_h_chr_pos=0:out_v_chr_pos=0,format=yuv420p -c:v libx265 -preset medium -x265-params crf=15:colorprim=bt709:transfer=bt709:colormatrix=bt709:master-display="G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,0)":max-cll="1000,400" -tag:v hvc1 -an {}'.format(w,h,fps,outyuv, os.path.join(savepath))
        cmd = 'ffmpeg -f rawvideo -vcodec rawvideo -s {}x{} -r {} -pix_fmt yuv420p -i {} -vf scale=out_color_matrix=bt709:out_h_chr_pos=0:out_v_chr_pos=0,format=yuv420p -c:v libx265 -preset medium -x265-params crf=20:colorprim=bt709:transfer=bt709:colormatrix=bt709:master-display="G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,0)":max-cll="1000,400" -tag:v hvc1 -an {}'.format(w,h,fps,outyuv, os.path.join(savepath))
    elif HDR_SDR == 'HDR':
        cmd = 'ffmpeg -f rawvideo -vcodec rawvideo -s {}x{} -r {} -pix_fmt yuv420p10 -i {} -vf scale=out_color_matrix=bt2020:out_h_chr_pos=0:out_v_chr_pos=0,format=yuv420p10le -c:v libx265 -preset medium -x265-params crf=15:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display="G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,0)":max-cll="1000,400" -tag:v hvc1 -an {}'.format(w,h,fps,outyuv, os.path.join(savepath))
    print(cmd)
    os.system(cmd)

