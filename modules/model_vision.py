import logging
import torch
import torch.nn as nn
from fastai.vision import *

from modules.attention import *
from modules.backbone import ResTranformer
from modules.model import Model
from modules.resnet import resnet45

import cv2
from torchvision import transforms
import math
from torch.nn import functional as F
#from easydict import EasyDict as edict
#import torch.distributed as dist
import numpy as np
from PIL import Image
from .utils import *
from . import torchutils



class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift

class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  







class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)
        self.thred = ifnone(config.thred, 0.7)

        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTranformer(config)
        else: self.backbone = resnet45()
        
        if config.model_vision_attention == 'position':
            mode = ifnone(config.model_vision_attention_mode, 'nearest')
            self.attention = PositionAttention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.model_vision_attention == 'attention':
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8*32,
            )
        else:
            raise Exception(f'{config.model_vision_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint)


        
        self.condnet = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))
        self.conv0 = nn.Conv2d(512, 64, 3, 1, 1)
        sft = []
        for i in range(16):
            sft.append(ResBlock_SFT())
        sft.append(SFTLayer())
        sft.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft = nn.Sequential(*sft)
        self.back = nn.Conv2d(64, 512, 3, 1, 1)

       
        self.project = nn.Linear(256, 26)

        self.classifier_6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1),  # fc6
            nn.ReLU(inplace=True)
        )
        self.exit_layer = nn.Conv2d(128, 1, kernel_size=1, padding=0)

        self.bce_logits_func = nn.CrossEntropyLoss()
        self.loss_func = nn.BCELoss()
        self.cos_similarity_func = nn.CosineSimilarity()
        self.triplelet_func = nn.TripletMarginLoss(margin=2.0)
        self.rf = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.rf2 = nn.Conv2d(1, 1, kernel_size=1, padding=0)


    def forward(self, images, mask, *args):
        
        features, outf, outside = self.backbone(images) 
        outf = F.upsample(outf, size=(32, 128), mode='bilinear')
        outside = F.upsample(outside, size=(32, 128), mode='bilinear')
        vec_pos = torch.sum(torch.sum(outf * mask, dim=3), dim=2) / torch.sum(mask) 
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3) 
        tmp_seg = self.cos_similarity_func(outf, vec_pos) 
        exit_feat_in = outside * tmp_seg.unsqueeze(dim=1) 
        outB_side_6 = self.classifier_6(exit_feat_in) 
        outB_side = self.exit_layer(outB_side_6) 

        segments = outB_side  
        segments = segments-segments.min()
        segments = segments / (segments.max()-segments.min())
        segments = torch.where(torch.isnan(segments), torch.zeros_like(segments), segments)

        mask2 = mask / mask.max() 
        segments = segments + mask2
        segments2 = torch.cat((segments, mask2), dim=1)
        segments2 = self.rf(segments2)
        segments2 = segments2 + mask2
        segments2 = self.rf2(segments2)


        e1 = torch.ones_like(segments2[segments2 > self.thred]).sum() if torch.ones_like(
            segments2[segments2 > self.thred]).sum() != 0. else 1.
        e2 = torch.ones_like(segments2[segments2 <= self.thred]).sum() if torch.ones_like(
            segments2[segments2 <= self.thred]).sum() != 0. else 1.
        lossre = torch.abs(segments2[segments2 > self.thred] - 1.0).sum() / e1 + torch.abs(
            segments2[segments2 <= self.thred] - 0.0).sum() / e2

        prfn8 = segments2.repeat(1, 8, 1, 1) 
        cond = self.condnet(prfn8)  
        fea = self.conv0(features)  
        res = self.sft((fea, cond)) 
        fea = fea + res             
        features = self.back(fea)  


        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs) # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths, 'lossre':lossre,
                'attn_scores': attn_scores, 'loss_weight':self.loss_weight, 'name': 'vision'}

