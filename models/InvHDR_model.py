import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.loss import Forw_strcture, Forw_orth
import numpy as np

logger = logging.getLogger('base')
class InvHDR_Model(BaseModel):
    def __init__(self, opt):
        super(InvHDR_Model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        # loss
        self.Forw_rec = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
        self.Forw_strcture = Forw_strcture()
        self.Forw_orth = Forw_orth()

        self.Forw_strcture = self.Forw_strcture.to(self.device)
        self.Forw_orth = self.Forw_orth.to(self.device)

        self.Back_rec = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
        self.Rec_back_cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # print network
        # self.print_network()
        self.load()

        self.total_time = 0
        self.time_ct = 0
        # for name, param in self.netG.module.projector_s.named_parameters():
        #     if param.requires_grad:
        #         print(name, param,'qq')

        if self.is_train:
            self.netG.train()

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()


    def feed_data(self, data):
        # self.ref_L = data['LQ'].to(self.device)  # LQ
        self.HDR_img = data['HDR_img'].to(self.device)  # GT
        self.SDR_img = data['SDR_img'].to(self.device)  # Noisy
        self.HDR_img_resize = data['HDR_img_resize'].to(self.device)
        self.S = data['S'].to(self.device)
        self.S = nn.functional.interpolate(self.S, size=(self.S.shape[2] // 2, self.S.shape[3] // 2), mode='bicubic', align_corners=True)

        self.SDR_imgnoise = self.gaussian_batch(
            [self.SDR_img.shape[0], 9, self.SDR_img.shape[2], self.SDR_img.shape[3]])

    def feed_data_test(self, data):
        self.HDR_img = data['HDR_img'].to(self.device)  # GT
        self.SDR_img = data['SDR_img'].to(self.device)  # Noisy
        self.S = data['S'].to(self.device)
        self.S = nn.functional.interpolate(self.S, size=(self.S.shape[2] // 2, self.S.shape[3] // 2), mode='bicubic', align_corners=True)

        self.SDR_imgnoise = self.gaussian_batch(
            [self.SDR_img.shape[0], 9, self.SDR_img.shape[2], self.SDR_img.shape[3]])

    def feed_test_data(self, data):
        self.noisy_H = data.to(self.device)

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y,  proj_s, proj_oc, proj_op, S):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Forw_rec(out, y)
        l_forw_s = self.train_opt['lambda_structure_forw'] * self.Forw_strcture(proj_s, S)
        l_forw_orth = self.train_opt['lambda_orth_forw'] * self.Forw_orth(proj_oc, proj_op)

        return l_forw_fit, l_forw_s, l_forw_orth

    def loss_backward(self, out, y,  proj_s, proj_oc, proj_op, S):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Back_rec(y, out)
        l_back_s = self.train_opt['lambda_structure_back'] * self.Forw_strcture(proj_s, S)
        l_back_orth = self.train_opt['lambda_orth_back'] * self.Forw_orth(proj_oc, proj_op)

        return l_back_rec, l_back_s, l_back_orth

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # forward
        self.SDR_pred, p1, c1, proj_s, proj_oc, proj_op, _, _ = self.netG(x=self.HDR_img)

        LR_ref = (self.SDR_img).detach()

        l_forw_fit, l_forw_s, l_forw_orth = self.loss_forward(self.SDR_pred[:, :3, :, :], LR_ref, proj_s, proj_oc, proj_op, self.S)

        # backward
        random_number = 0.2*np.random.randn()
        random_number = np.abs(random_number)
        random_number = np.clip(random_number, 0, 1)
        y_ = random_number * self.SDR_pred + (1-random_number) * self.SDR_img[:, :3, :, :]

        self.HDR_pred, p2, c2, proj_s, proj_oc, proj_op, _, _ = self.netG(x=y_, rev=True)

        l_back_rec, l_back_s, l_back_orth = self.loss_backward(self.HDR_pred, self.HDR_img,  proj_s, proj_oc, proj_op, self.S)

        # total loss
        loss = l_forw_fit  + l_forw_s +l_forw_orth + \
                l_back_rec  + l_back_s + l_back_orth

        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_forw_s'] = l_forw_s.item()
        self.log_dict['l_forw_orth'] = l_forw_orth.item()

        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['l_back_s'] = l_back_s.item()
        self.log_dict['l_back_orth'] = l_back_orth.item()

    def test(self, self_ensemble=False):
        self.input = self.SDR_img

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        with torch.no_grad():
            if self_ensemble:
                forward_function = self.netG.forward
                self.fake_H = self.forward_x8(self.input, forward_function, gaussian_scale)
            else:
                y_forw = self.input[:, :3, :, :]

                self.fake_H, p1, c1, proj_s, proj_oc, proj_op, HF, LF = self.netG(x=y_forw, rev=True)

        self.netG.train()


    def test_bi(self):
        self.input = self.SDR_img
        self.netG.eval()
        with torch.no_grad():
            output, self.p1, self.c1, self.proj_s_HDR, self.proj_oc_HDR, self.proj_op_HDR, self.HF_HDR, self.LF_HDR = self.netG(x=self.HDR_img)
            self.fake_H_bi, self.p2, self.c2, self.proj_s_SDR, self.proj_oc_SDR, self.proj_op_SDR, self.HF_SDR, self.LF_SDR = self.netG(x=output, rev=True)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['HDR_pred'] = self.fake_H.detach()[0].float().cpu()
        out_dict['HDR_pred_bi'] = self.fake_H_bi.detach()[0].float().cpu()
        out_dict['HDR_img'] = self.HDR_img.detach()[0].float().cpu()
        return out_dict

    def get_current_visuals_gpu(self):
        out_dict = OrderedDict()
        # out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['HDR_pred'] = self.fake_H.detach()[0].float()
        out_dict['SDR_img'] = self.SDR_img.detach()[0].float()
        out_dict['HDR_img'] = self.HDR_img.detach()[0].float()
        return out_dict
    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):

        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
