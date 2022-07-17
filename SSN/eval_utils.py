import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# cudnn.enabled = True
# cudnn.benchmark = True
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import os.path as osp
import sys
import numpy as np
import pprint
import timeit
import time
import PIL
import copy
from easydict import EasyDict as edict

# from lib import utils
from lib import nputils
from lib import torchutils

from configs.cfg_dataset import \
    cfg_textseg, cfg_cocots, cfg_mlt, cfg_icdar13, cfg_totaltext
from configs.cfg_model import cfg_texrnet as cfg_mdel

from lib.cfg_helper import cfg_unique_holder as cfguh
from lib.cfg_helper import experiment_folder, set_debug_cfg
from lib.cfg_helper import common_argparse, common_initiates

from lib.data_factory import \
    get_dataset, collate, \
    get_loader, get_transform, \
    get_formatter, DistributedSampler

from lib.log_service import print_log, torch_to_numpy
from lib import evaluate_service as eva
from train_utils import exec_container

cfguh().add_code(osp.basename(__file__))

import os
import cv2
#from data.data_utils import Augmenter

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import lmdb
import six
#from fastai.vision import *
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import random, re, warnings

class listDataset(Dataset):
    def __init__(self, imgdir=None, list_file=None, transform=None, inTrain=False, p_aug=0, vert_test=False):

        self.list_file = list_file

        self.transform = transform
        self.imgdir = imgdir
        self.inTrain = inTrain
        self.p_aug = p_aug
        self.vert_test = vert_test

        #if inTrain:
        #    self.aug = Augmenter(p=self.p_aug)

        self.convert_mode = 'RGB'
        self.check_length = True
        self.max_length = 25
        self.env = lmdb.open(str(self.list_file), readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))


    def __len__(self):
        #return self.nSamples
        #print(self.length)
        return self.length

    def _next_image(self, index):
        next_index = index+1
        #next_index = random.randint(0, len(self) - 1)
        return self.get(next_index)

    def _check_image(self, x, pixels=6):
        if x.size[0] <= pixels or x.size[1] <= pixels: return False
        else: return True
        #return True

    def get(self, index):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{index+1:09d}', f'label-{index+1:09d}'
            try:
                label = str(txn.get(label_key.encode()), 'utf-8')  # label
                label = re.sub('[^0-9a-zA-Z]+', '', label)
                #if self.check_length and self.max_length > 0:
                #    if len(label) >= self.max_length or len(label) <= 0:
                        #logging.info(f'Long or short text image is found: {self.name}, {idx}, {label}, {len(label)}')
                #        return self._next_image(index)
                label = label[:self.max_length]
                label = label.lower()

                imgbuf = txn.get(image_key.encode())  # image
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
                    image = PIL.Image.open(buf).convert(self.convert_mode)
                    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                #if self.inTrain and not self._check_image(image):
                    #logging.info(f'Invalid image is found: {self.name}, {idx}, {label}, {len(label)}')
                #    return self._next_image(index)

                h, w, _ = image.shape
                #if min(h, w)<=5:
                #    return self._next_image(index)
            except:
                import traceback
                traceback.print_exc()
                #logging.info(f'Corrupted image is found: {self.name}, {index}, {label}, {len(label)}')
                return self._next_image(index)
            return image, label, index

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        image, label, idx_new = self.get(index)

        h, w, _ = image.shape

        image = image.astype(np.float32) / 255  # image
        #nputils.normalize_2d(self.mean, self.std)(data.astype(np.float32))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = np.transpose(image, (2, 0, 1))

        for j in range(3):
            image[j] = (image[j]  - mean[j]) /  std[j]

        image = image.astype(np.float32)
        
        return (image, label, idx_new)






class eval(exec_container):
    def prepare_dataloader(self):
        
        cfg = cfguh().cfg
        train_list = [
            #'/../data_real/evaluation/benchmark/CUTE80', ]
            #'/../data_real/evaluation/benchmark/IC13_1015',]
            #'/../data_real/evaluation/benchmark/IC15_2077',]
            #'/../data_real/evaluation/benchmark/IIIT5k_3000',]
            #'/../data_real/evaluation/benchmark/SVT',]
            #'/../data_real/evaluation/benchmark/SVTP',]
            #'/../data_lmdb_release/evaluation/IIIT5k_3000',]
            #'/../data_lmdb_release/evaluation/SVT', ]
            #'/../data_lmdb_release/evaluation/SVTP',]
            #'/../data_lmdb_release/evaluation/IC13_857',]
            '/../data_lmdb_release/evaluation/IC15_1811',]
            #'/../data_lmdb_release/evaluation/CUTE80'],
            #'/../data_lmdb_release/evaluation/SVTP'],

        data_loader_list = []
        dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        dataset_list = []
        for d in train_list:
            _batch_size = max(round(cfg.TEST.BATCH_SIZE), 1)
            dataset = listDataset(  # imgdir=configs.image_dir,
                list_file=d,
                #transform=transforms,
                #formatter=formatter,
                inTrain=False,)
            #
            dataset_list.append(dataset)
            # dataset_list.append(dataset1)
        _dataset = ConcatDataset(dataset_list)
        # batch_size_list.append(str(_batch_size))
        # Total_batch_size += _batch_size
        sampler = DistributedSampler(_dataset, shuffle=False, extend=True)
        evalloader = torch.utils.data.DataLoader(_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=cfg.DATA.NUM_WORKERS_PER_GPU,
                                  drop_last=False, pin_memory=False,
                                  collate_fn=collate(),
                                                   )
        ##"""
        return {
            'dataloader' : evalloader,}

def set_cfg(cfg, dsname):
    cfg.CUDA = True

    cfg.DATA.DATASET_MODE = '>>>>later<<<<'
    cfg.DATA.LOADER_PIPELINE = [
        'NumpyImageLoader',
        #'Icdar13_CharBboxLoader_Original']
        #'Icdar13_SeglabelLoader_Original']
        'NumpySeglabelLoader'] #,
        #'CharBboxSpLoader']
    cfg.DATA.ALIGN_CORNERS = True
    cfg.DATA.IGNORE_LABEL = cfg.DATA.SEGLABEL_IGNORE_LABEL

    cfg.DATA.TRANS_PIPELINE = [
        'UniformNumpyType',
        'NormalizeUint8ToZeroOne', 
        'Normalize',]

    cfg.DATA.FORMATTER = 'SemanticFormatter' 
    cfg.DATA.EFFECTIVE_CLASS_NUM = cfg.DATA.CLASS_NUM

    if dsname == 'textseg':
        cfg.DATA.LOADER_DERIVED_CLS_MAP_TO = 'bg'
        cfg.DATA.LOADER_PIPELINE = [
            'NumpyImageLoader', 
            'TextSeg_SeglabelLoader']
    elif dsname == 'cocots':
        pass
    elif dsname == 'mlt':
        cfg.DATA.LOADER_PIPELINE = [
            'NumpyImageLoader', 
            'Mlt_SeglabelLoader']
    elif dsname == 'icdar13':
        pass
    elif dsname == 'totaltext':
        cfg.DATA.LOADER_PIPELINE = [
            'NumpyImageLoader', 
            'TotalText_SeglabelLoader']
    elif dsname == 'textssc':
        pass
    else:
        raise ValueError

    ##########
    # resnet #
    ##########
    cfg.MODEL.RESNET.MODEL_TAGS = ['base', 'dilated', 'resnet101', 'os16']
    cfg.MODEL.RESNET.CONV_TYPE = 'conv'
    cfg.MODEL.RESNET.BN_TYPE = ['bn', 'syncbn'][0]
    cfg.MODEL.RESNET.RELU_TYPE = 'relu'
    cfg.MODEL.RESNET.USE_MAXPOOL = True

    ###########
    # deeplab #
    ###########
    cfg.MODEL.DEEPLAB.MODEL_TAGS = ['resnet', 'v3+', 'os16', 'base']
    cfg.MODEL.DEEPLAB.OUTPUT_CHANNEL_NUM = 256
    cfg.MODEL.DEEPLAB.CONV_TYPE = cfg.MODEL.RESNET.CONV_TYPE
    cfg.MODEL.DEEPLAB.BN_TYPE = cfg.MODEL.RESNET.BN_TYPE
    cfg.MODEL.DEEPLAB.RELU_TYPE = cfg.MODEL.RESNET.RELU_TYPE
    cfg.MODEL.DEEPLAB.ASPP_WITH_GAP = True

    cfg.MODEL.DEEPLAB.FREEZE_BACKBONE_BN = False
    cfg.MODEL.DEEPLAB.INTERPOLATE_ALIGN_CORNERS = \
        cfg.DATA.ALIGN_CORNERS

    ###########
    # texrnet #
    ###########
    cfg.MODEL.TEXRNET.MODEL_TAGS = ['deeplab']
    cfg.MODEL.TEXRNET.PRETRAINED_PTH = None
    cfg.MODEL.TEXRNET.INPUT_CHANNEL_NUM = \
        cfg.MODEL.DEEPLAB.OUTPUT_CHANNEL_NUM
    cfg.MODEL.TEXRNET.SEMANTIC_CLASS_NUM = \
        cfg.DATA.EFFECTIVE_CLASS_NUM
    cfg.MODEL.TEXRNET.REFINEMENT_CHANNEL_NUM = [
        3+cfg.MODEL.DEEPLAB.OUTPUT_CHANNEL_NUM
        +cfg.DATA.EFFECTIVE_CLASS_NUM, 64, 64]

    cfg.MODEL.TEXRNET.CONV_TYPE = cfg.MODEL.RESNET.CONV_TYPE
    cfg.MODEL.TEXRNET.BN_TYPE = cfg.MODEL.RESNET.BN_TYPE
    cfg.MODEL.TEXRNET.RELU_TYPE = cfg.MODEL.RESNET.RELU_TYPE

    cfg.MODEL.TEXRNET.ALIGN_CORNERS = cfg.DATA.ALIGN_CORNERS    
    cfg.MODEL.TEXRNET.INIT_BIAS_ATTENTION_WITH = None
    cfg.MODEL.TEXRNET.BIAS_ATTENTION_TYPE = 'cossim'
    cfg.MODEL.TEXRNET.INEVAL_OUTPUT_ARGMAX = False

    ###########
    # general #
    ###########
    cfg.DATA.NUM_WORKERS_PER_GPU = 0 #1

    cfg.TEST.BATCH_SIZE_PER_GPU = 1
    cfg.TEST.DISPLAY = 10
    cfg.TEST.VISUAL = False
    cfg.TEST.SUB_DIR = '>>>>later<<<<'
    cfg.TEST.OUTPUT_RESULT = True #False

    cfg.TEST.INFERENCE_FLIP = False
    cfg.TEST.INFERENCE_MS = [
        ['0.75x', int(512*0.75)+1],
        ['1.00x', int(512*1.00)+1],
        ['1.25x', int(512*1.25)+1],
        ['1.50x', int(512*1.50)+1],
        ['1.75x', int(512*1.75)+1],
        ['2.00x', int(512*2.00)+1],   
        ['2.25x', int(512*2.25)+1],   
        ['2.50x', int(512*2.50)+1],   
    ]
    cfg.TEST.INFERENCE_MS_ALIGN_CORNERS = \
        cfg.DATA.ALIGN_CORNERS
    cfg.TEST.FIND_N_WORST = 100

    if cfg.DATA.DATASET_NAME not in [dsname]:
        raise ValueError
    if cfg.MODEL.MODEL_NAME not in ['texrnet']:
        raise ValueError
    return cfg

def set_cfg_hrnetw48(cfg):
    try:
        cfg.MODEL.pop('DEEPLAB')
    except:
        pass
    try:
        cfg.MODEL.pop('RESNET')
    except:
        pass

    cfg.MODEL.HRNET = edict()
    cfg.MODEL.HRNET.MODEL_TAGS = ['v0', 'base']
    cfg.MODEL.HRNET.PRETRAINED_PTH = None 

    cfg.MODEL.HRNET.STAGE1_PARA = {
        'NUM_MODULES'  : 1,
        'NUM_BRANCHES' : 1,
        'BLOCK'        : 'BOTTLENECK',
        'NUM_BLOCKS'   : [4],
        'NUM_CHANNELS' : [64],
        'FUSE_METHOD'  : 'SUM',}
    cfg.MODEL.HRNET.STAGE2_PARA = {
        'NUM_MODULES'  : 1,
        'NUM_BRANCHES' : 2,
        'BLOCK'        : 'BASIC',
        'NUM_BLOCKS'   : [4, 4],
        'NUM_CHANNELS' : [48, 96],
        'FUSE_METHOD'  : 'SUM',}
    cfg.MODEL.HRNET.STAGE3_PARA = {
        'NUM_MODULES'  : 4,
        'NUM_BRANCHES' : 3,
        'BLOCK'        : 'BASIC',
        'NUM_BLOCKS'   : [4, 4, 4],
        'NUM_CHANNELS' : [48, 96, 192],
        'FUSE_METHOD'  : 'SUM',}
    cfg.MODEL.HRNET.STAGE4_PARA = {
        'NUM_MODULES'  : 3,
        'NUM_BRANCHES' : 4,
        'BLOCK'        : 'BASIC',
        'NUM_BLOCKS'   : [4, 4, 4, 4],
        'NUM_CHANNELS' : [48, 96, 192, 384],
        'FUSE_METHOD'  : 'SUM',}
    cfg.MODEL.HRNET.FINAL_CONV_KERNEL = 1

    cfg.MODEL.HRNET.OUTPUT_CHANNEL_NUM = sum([48, 96, 192, 384])
    cfg.MODEL.HRNET.ALIGN_CORNERS = \
        cfg.DATA.ALIGN_CORNERS
    cfg.MODEL.HRNET.IGNORE_LABEL = \
        cfg.DATA.IGNORE_LABEL
    cfg.MODEL.HRNET.BN_MOMENTUM = 'hardcoded to 0.1'
    cfg.MODEL.HRNET.LOSS_TYPE = 'ce'
    cfg.MODEL.HRNET.INTRAIN_GETPRED = False

    ###########
    # TEXRNET #
    ###########
    cfg.MODEL.TEXRNET.MODEL_TAGS = ['hrnet']
    cfg.MODEL.TEXRNET.INPUT_CHANNEL_NUM = \
        cfg.MODEL.HRNET.OUTPUT_CHANNEL_NUM
    cfg.MODEL.TEXRNET.REFINEMENT_CHANNEL_NUM = [
        3+cfg.MODEL.HRNET.OUTPUT_CHANNEL_NUM
        +cfg.DATA.EFFECTIVE_CLASS_NUM, 64, 64]
    cfg.MODEL.TEXRNET.CONV_TYPE = 'conv'
    cfg.MODEL.TEXRNET.BN_TYPE = 'bn'
    cfg.MODEL.TEXRNET.RELU_TYPE = 'relu'
    return cfg

class es(object):
    def __init__(self):
        super().__init__()

    def output_f(self, item):
        outdir = osp.join(
            cfguh().cfg.LOG_DIR, 'result')
        if not osp.exists(outdir):
            os.makedirs(outdir)
        outformat = osp.join(outdir, '{}_{}_{}_{}.png')

        for i, fni in enumerate(item['fn']):
            h, w = item['im'].shape[2], item['im'].shape[3]
            #p = (item['prfn'][i]*255).astype(np.uint8)
            p = (item['prfnc'][0][1] * 255).astype(np.uint8)
            label = item['gtsem'][0]
            print(int(fni.item()))
            PIL.Image.fromarray(p).save(outformat.format(int(fni.item()), label, h, w))

    def visual_f(self, item):
        pass

    def main(self,
             RANK,
             batch,
             net,
             **kwargs):
        cfg = cfguh().cfg
        ac = cfg.TEST.INFERENCE_MS_ALIGN_CORNERS

        im, gtsem, fn = batch
        #print(fn)

        bs, _, oh, ow = im.shape

        if cfg.CUDA:
            im = im.to(RANK)

        # ms-flip inference
        psemc_ms, prfnc_ms, pcount_ms = {}, {}, {}
        pattkey, patt = {}, {}
        for mstag, mssize in cfg.TEST.INFERENCE_MS:
            # by area
            ratio = np.sqrt(mssize**2 / (oh*ow))
            th, tw = int(oh*ratio), int(ow*ratio)
            tw = tw//32*32+1
            th = th//32*32+1

            imi = {
                'nofp' : torchutils.interpolate_2d(
                    size=(th, tw), mode='bilinear', 
                    align_corners=ac)(im)}                    
            if cfg.TEST.INFERENCE_FLIP:
                imi['flip'] = torch.flip(imi['nofp'], dims=[-1])

            for fliptag, imii in imi.items():
                with torch.no_grad():
                    pred = net(imii)
                    psem = torchutils.interpolate_2d(
                        size=(oh, ow), 
                        mode='bilinear', align_corners=ac)(pred['predsem']) 
                    prfn = torchutils.interpolate_2d(
                        size=(oh, ow), 
                        mode='bilinear', align_corners=ac)(pred['predrfn']) 

                    if fliptag == 'flip':
                        psem = torch.flip(psem, dims=[-1])
                        prfn = torch.flip(prfn, dims=[-1])
                    elif fliptag == 'nofp':
                        pass
                    else:
                        raise ValueError
                
                try:
                    psemc_ms[mstag]  += psem
                    prfnc_ms[mstag]  += prfn
                    pcount_ms[mstag] += 1
                except:
                    psemc_ms[mstag]  = psem
                    prfnc_ms[mstag]  = prfn
                    pcount_ms[mstag] = 1

            # if flip, this is the attention that flipped.
            try:
                pattkey[mstag] = pred['att_key']
            except:
                pattkey[mstag] = None
            try:
                patt[mstag] = pred['att']
            except:
                patt[mstag] = None

        predc = []
        for predci in [psemc_ms, prfnc_ms]:        
            p = sum([pi for pi in predci.values()])
            p /= sum([ni for ni in pcount_ms.values()])
            predc.append(p)
            p = {ki:pi/pcount_ms[ki] for ki, pi in predci.items()}
            predc.append(p)
        psemc, psemc_ms, prfnc, prfnc_ms = predc     #soft prediction is prfnc
        
        psem = torch.argmax(psemc, dim=1)
        prfn = torch.argmax(prfnc, dim=1)
        im, gtsem, psemc, psemc_ms, psem, prfnc, prfnc_ms, prfn = \
            torch_to_numpy(
                im, gtsem, psemc, psemc_ms, psem, prfnc, prfnc_ms, prfn)
        pattkey, patt = torch_to_numpy(pattkey, patt)

        return {
            'im'    : im,
            'gtsem' : gtsem, 
            'psem'  : psem,
            'psemc' : psemc, 
            'psemc_ms' : psemc_ms,
            'prfn'  : prfn,
            'prfnc' : prfnc,
            'prfnc_ms' : prfnc_ms,
            'pattkey': pattkey,
            'patt'  : patt,
            'fn'    : fn, }

    def __call__(self, 
                 RANK,
                 dataloader,
                 net,
                 **paras):
        cfg = cfguh().cfg
        evaluator = eva.distributed_evaluator(
            name=['rfn'], 
            sample_n=len(dataloader.dataset))

        time_check = timeit.default_timer()

        for idx, batch in enumerate(dataloader):
            item = self.main(
                RANK=RANK, 
                batch=batch, 
                net=net,
                **paras)
            #gtsem, prfn = [item[i] for i in [
            #    'gtsem', 'prfn']]
            gtsem, prfnc = [item[i] for i in [
                'gtsem', 'prfnc']]

            if cfg.TEST.OUTPUT_RESULT:
                self.output_f(item)
