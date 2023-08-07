"""
This file includes code derived from the DDPM-CD project,
which is licensed under the MIT License. The original license notice
is included below:

MIT License

Copyright (c) 2023 Chaminda Bandara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import lightning as pl
import torch
import torch.nn as nn
from .model import networks as networks
from collections import OrderedDict
from ..utils import download_and_verify
import os
from torchmetrics.classification import f_beta
from collections import OrderedDict
from .misc.metric_tools import ConfuseMatrixMeter
from .misc.metric_tools import ConfuseMatrixMeter
from .misc.torchutils import get_scheduler
from . import data as Data
from . import model as Model

class DDPM(pl.LightningModule):
    def __init__(self, opt):
        super(DDPM, self).__init__()
        self.opt = opt

        if self.opt['model']['diffusion']['download']:
            download_dir_path = self.opt['model']['diffusion']['download_dir']
            self.opt_file = os.path.join(download_dir_path, "I190000_E97_opt.pth")
            self.gen_file = os.path.join(download_dir_path, "I190000_E97_gen.pth")
            
            os.makedirs(download_dir_path, exist_ok=True)

            if not os.path.isfile(self.opt_file):
                assert download_and_verify(self.opt['model']['diffusion']['opt_url'], self.opt_file, self.opt['model']['diffusion']['opt_hash'])
            
            if not os.path.isfile(self.gen_file):
                assert download_and_verify(self.opt['model']['diffusion']['gen_url'], self.gen_file, self.opt['model']['diffusion']['gen_hash'])
        else:
            self.opt_file = self.opt['path']['resume_state'] + '_opt.pth'
            self.gen_file = self.opt['path']['resume_state'] + '_gen.pth'

            assert os.path.isfile(self.opt_file)
            assert os.path.isfile(self.gen_file)

        self.netG = networks.define_G(opt)
        self.schedule_phase = None
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        self.log_vars = OrderedDict()
        if self.opt['phase'] == 'train':
            self.netG.train()
        self.optG = self.configure_optimizers()

    def forward(self, x):
        return self.netG(x)

    def training_step(self, batch, batch_idx):
        self.data = batch
        self.optG.zero_grad()

        l_pix = self(batch)
        b, c, h, w = batch['img'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)

        self.log_vars['l_pix'] = l_pix.item()
        self.log('train_loss', l_pix)
        return l_pix

    def validation_step(self, batch, batch_idx):
        self.set_new_noise_schedule(self.opt['model']['beta_schedule']['val'], schedule_phase='val')
        in_channels = self.opt['model']['unet']['in_channel']
        img_size = self.opt['datasets']['val']['resolution']
        self.sampled_img = self.netG.sampling_imgs(in_channels, img_size, continous=False)
        self.set_new_noise_schedule(self.opt['model']['beta_schedule']['train'], schedule_phase='train')
        
        l_pix = self(batch)
        b, c, h, w = batch['img'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        # self.log_vars['l_pix'] = l_pix.item()
        self.log('val_loss', l_pix)

        return l_pix

    def configure_optimizers(self):
        if self.opt['model']['finetune_norm']:
            optim_params = []
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                if k.find('transformer') >= 0:
                    v.requires_grad = True
                    v.data.zero_()
                    optim_params.append(v)
        else:
            optim_params = list(self.netG.parameters())

        if self.opt['train']["optimizer"]["type"] == "adam":
            self.optG = torch.optim.Adam(optim_params, lr=self.opt['train']["optimizer"]["lr"])
        elif self.opt['train']["optimizer"]["type"] == "adamw":
            self.optG = torch.optim.AdamW(optim_params, lr=self.opt['train']["optimizer"]["lr"])
        else:
            raise NotImplementedError('Optimizer [{:s}] not implemented'.format(self.opt['train']["optimizer"]["type"]))
        return self.optG
    
    def _split_images(self, images, bands='rgb'):
        n_bands = 3 if bands == "rgb" else 13
        pre, post = images[:, 0:n_bands], images[:, n_bands:2*n_bands]
        return pre, post

    # Get feature representations for a given image
    def get_feats(self, batch, t):
        self.netG.eval()
        A, B = self._split_images(batch['image'])
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                fe_A, fd_A = self.netG.module.feats(A, t)
                fe_B, fd_B = self.netG.module.feats(B, t)
            else:
                fe_A, fd_A = self.netG.feats(A, t)
                fe_B, fd_B = self.netG.feats(B, t)
        self.netG.train()
        return fe_A, fd_A, fe_B, fd_B

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.sampled_img = self.netG.module.sample(batch_size, continous)
            else:
                self.sampled_img = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_vars

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['SAM'] = self.sampled_img.detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            # network.load_state_dict(torch.load(
            #     self.gen_file, 
            #     map_location=self.device), 
            #     strict=(not self.opt['model']['finetune_norm']))
            print(f'Loading DDPM gen state dict from {self.gen_file}')
            network.load_state_dict(torch.load(
                self.gen_file, 
                map_location=self.device), 
                strict=True)
                
            if self.opt['phase'] == 'train' and self.opt['path']['resume_opt']:
                print(f'Loading DDPM opt state dict from {self.opt_file}')
                #optimizer
                opt = torch.load(self.opt_file, map_location=self.device)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

class CD(pl.LightningModule):
    def __init__(self, opt):
        super(CD, self).__init__()
        self.diffusion_model = DDPM(opt)

        # Set noise schedule for the diffusion model
        self.diffusion_model.set_new_noise_schedule(
            opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
        
        self.opt = opt
        self.batch_size = opt['datasets']['train']['batch_size']
        self.diffusion_model.eval()  # Freeze the diffusion model
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

        # define network and load pretrained models
        self.netCD = networks.define_CD(opt)

        # set loss and load resume state
        self.loss_type = opt['model_cd']['loss_type']
        if self.loss_type == 'ce':
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplementedError()

        self.train_vars = OrderedDict()
        self.val_vars = OrderedDict()
        self.test_vars = OrderedDict()

        self.train_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        self.val_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        self.test_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        
        self.len_train_dataloader = opt["len_train_dataloader"]
        self.len_val_dataloader = opt["len_val_dataloader"]
        self.exp_lr_scheduler_netCD = None

    def forward(self, batch):
        # Feeding data to diffusion model and get features
        f_A, f_B = self.get_diffusion_feats(batch)
        # Feeding features from the diffusion model to the CD model
        self.feed_data(f_A, f_B, batch)
        return self.netCD(f_A, f_B)

    def get_diffusion_feats(self, batch):
        # self.diffusion_model.feed_data(batch)
        f_A=[]
        f_B=[]
        for t in self.opt['model_cd']['t']:
            fe_A_t, fd_A_t, fe_B_t, fd_B_t = self.diffusion_model.get_feats(batch, t=t) #np.random.randint(low=2, high=8)
            if self.opt['model_cd']['feat_type'] == "dec":
                f_A.append(fd_A_t)
                f_B.append(fd_B_t)
                # Uncommet the following line to visualize features from the diffusion model
                # for level in range(0, len(fd_A_t)):
                #     print_feats(opt=opt, train_data=train_data, feats_A=fd_A_t, feats_B=fd_B_t, level=level, t=t)
                # del fe_A_t, fe_B_t
            else:
                f_A.append(fe_A_t)
                f_B.append(fe_B_t)
                del fd_A_t, fd_B_t

        return f_A, f_B

    def training_step(self, batch, batch_idx):
        self.pred_cm = self.forward(batch)
        l_cd = self.loss_func(self.pred_cm, batch["mask"].long())
        self._collect_running_batch_states(self.train_metric, self.train_vars)
        self.log('train_loss', l_cd)
        return l_cd
    
    def on_train_epoch_end(self):
        self._collect_epoch_states(self.train_metric, self.train_vars)
        logs = self.train_vars
        self.log_dict({
            'training/mF1': logs['epoch_acc'],
            'training/mIoU': logs['miou'],
            'training/OA': logs['acc'],
            'training/change-F1': logs['F1_1'],
            'training/no-change-F1': logs['F1_0'],
            'training/change-IoU': logs['iou_1'],
            'training/no-change-IoU': logs['iou_0'],
        })
        self._clear_cache(self.train_metric)

    def validation_step(self, batch, batch_idx):
        self.pred_cm = self(batch)
        l_cd = self.loss_func(self.pred_cm, batch["mask"].long())
        self._collect_running_batch_states(self.val_metric, self.val_vars)
        self.log('val_loss', l_cd)
        return l_cd

    def on_validation_epoch_end(self):
        self._collect_epoch_states(self.val_metric, self.val_vars)
        logs = self.val_vars
        self.log_dict({
            'validation/mF1': logs['epoch_acc'],
            'validation/mIoU': logs['miou'],
            'validation/OA': logs['acc'],
            'validation/change-F1': logs['F1_1'],
            'validation/no-change-F1': logs['F1_0'],
            'validation/change-IoU': logs['iou_1'],
            'validation/no-change-IoU': logs['iou_0'],
        })
        self._clear_cache(self.val_metric)

    def test_step(self, batch, batch_idx):
        self.pred_cm = self(batch)
        l_cd = self.loss_func(self.pred_cm, batch["mask"].long())
        self._collect_running_batch_states(self.test_metric, self.test_vars)
        self.log('test_loss', l_cd)
        return l_cd

    def on_test_epoch_end(self):
        self._collect_epoch_states(self.test_metric, self.test_vars)
        logs = self.test_vars
        self.log_dict({
            'test/mF1': logs['epoch_acc'],
            'test/mIoU': logs['miou'],
            'test/OA': logs['acc'],
            'test/change-F1': logs['F1_1'],
            'test/no-change-F1': logs['F1_0'],
            'test/change-IoU': logs['iou_1'],
            'test/no-change-IoU': logs['iou_0'],
        })
        self._clear_cache(self.test_metric)

    def configure_optimizers(self):
        optim_cd_params = list(self.netCD.parameters())

        if self.opt['train']["optimizer"]["type"] == "adam":
            self.optCD = torch.optim.Adam(
                optim_cd_params, lr=self.opt['train']["optimizer"]["lr"])
        elif self.opt['train']["optimizer"]["type"] == "adamw":
            self.optCD = torch.optim.AdamW(
                optim_cd_params, lr=self.opt['train']["optimizer"]["lr"])
        else:
            raise NotImplementedError(
                'Optimizer [{:s}] not implemented'.format(self.opt['train']["optimizer"]["type"]))

        if self.exp_lr_scheduler_netCD is None:
            self.exp_lr_scheduler_netCD = get_scheduler(optimizer=self.optCD, args=self.opt['train'])

        return [self.optCD], [self.exp_lr_scheduler_netCD]

    # Feeding all data to the CD model
    def feed_data(self, feats_A, feats_B, data):
        self.feats_A = feats_A
        self.feats_B = feats_B
        self.data = data

    # Get current visuals
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred_cm'] = torch.argmax(self.pred_cm, dim=1, keepdim=False)
        out_dict['gt_cm'] = self.data['mask']
        return out_dict

    # Printing the CD network
    def print_network(self):
        s, n = self.get_network_description(self.netCD)
        if isinstance(self.netCD, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netCD.__class__.__name__,
                                             self.netCD.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netCD.__class__.__name__)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['scheduler'] = self.exp_lr_scheduler_netCD.state_dict()
        checkpoint['state_dict'] = {k: v for k, v in self.state_dict().items() if 'diffusion_model' not in k}

        # Call the base implementation to ensure any additional behavior is executed
        return checkpoint

    def load_state_dict(self, state_dict, strict=True):
        # Make sure to call the super().load_state_dict() function with strict=False to prevent it from throwing an error
        super().load_state_dict(state_dict, strict=False)
        self.diffusion_model = DDPM(self.opt)
        self.diffusion_model.load_network()
        self.diffusion_model.eval()  # Freeze the diffusion model
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

    # def on_load_checkpoint(self, checkpoint):
    #     if 'scheduler' in checkpoint:
    #         self.configure_optimizers()
    #         self.exp_lr_scheduler_netCD.load_state_dict(checkpoint['scheduler'])
    # Saving the network parameters
    # def save_network(self, epoch, is_best_model = False):
    #     cd_gen_path = os.path.join(
    #         self.opt['path']['checkpoint'], 'cd_model_E{}_gen.pth'.format(epoch))
    #     cd_opt_path = os.path.join(
    #         self.opt['path']['checkpoint'], 'cd_model_E{}_opt.pth'.format(epoch))
        
    #     if is_best_model:
    #         best_cd_gen_path = os.path.join(
    #             self.opt['path']['checkpoint'], 'best_cd_model_gen.pth'.format(epoch))
    #         best_cd_opt_path = os.path.join(
    #             self.opt['path']['checkpoint'], 'best_cd_model_opt.pth'.format(epoch))

    #     # Save CD model pareamters
    #     network = self.netCD
    #     if isinstance(self.netCD, nn.DataParallel):
    #         network = network.module
    #     state_dict = network.state_dict()
    #     for key, param in state_dict.items():
    #         state_dict[key] = param.cpu()
    #     torch.save(state_dict, cd_gen_path)
    #     if is_best_model:
    #         torch.save(state_dict, best_cd_gen_path)

    #     # Save CD optimizer paramers
    #     opt_state = {'epoch': epoch,
    #                  'scheduler': None, 
    #                  'optimizer': None}
    #     opt_state['optimizer'] = self.optCD.state_dict()
    #     torch.save(opt_state, cd_opt_path)
    #     if is_best_model:
    #         torch.save(opt_state, best_cd_opt_path)

    # Loading pre-trained CD network
    def load_network(self):
        load_path = self.opt['path_cd']['finetune_path']
        if load_path is not None:
            gen_path = os.path.join(load_path, 'best_cd_model_gen.pth')
            opt_path = os.path.join(load_path, 'best_cd_model_opt.pth')
            
            # change detection model
            network = self.netCD
            if isinstance(self.netCD, nn.DataParallel):
                network = network.module
            print(f'Loading CD gen state dict from {gen_path}.')
            network.load_state_dict(torch.load(
                gen_path), strict=True)
            
            if self.opt['phase'] == 'train' and self.opt['path_cd']['resume_opt']:
                print(f'Loading CD opt state dict from {opt_path}.') 
                opt = torch.load(opt_path)
                self.optCD.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
    
    # Functions related to computing performance metrics for CD
    def _update_metric(self, metric):
        """
        update metric
        """
        G_pred = self.pred_cm.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = metric.update_cm(pr=G_pred.cpu().numpy(), gt=self.data['mask'].detach().cpu().numpy())
        return current_score
    
    # Collecting status of the current running batch
    def _collect_running_batch_states(self, metric, log_vars):
        self.running_acc = self._update_metric(metric)
        log_vars['running_acc'] = self.running_acc.item()

    # Collect the status of the epoch
    def _collect_epoch_states(self, metric, log_vars):
        scores = metric.get_scores()
        self.epoch_acc = scores['mf1']
        log_vars['epoch_acc'] = self.epoch_acc.item()

        for k, v in scores.items():
            log_vars[k] = v

    # Rest all the performance metrics
    def _clear_cache(self, metric):
        metric.clear()

    # Finctions related to learning rate sheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netCD.step()