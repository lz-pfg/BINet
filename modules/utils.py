import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


import numpy as np
import copy

import matplotlib.pyplot as plt

from . import nputils
from . import torchutils


def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_component(object):
    """
    The singleton class that can
        register a compnent and
        get small components
    """
    def __init__(self):
        self.component = {}
        self.register('none', None)

        # general convolution
        self.register(
            'conv', nn.Conv2d,
            kwinit={
                'bias'        : False}, )
        self.register(
            'conv1x1', nn.Conv2d,
            kwmap={
                'kernel_size' : lambda x:1,
                'padding'     : lambda x:0,},
            kwinit={
                'bias'        : False}, )
        self.register(
            'conv3x3', nn.Conv2d,
            kwmap={
                'kernel_size' : lambda x:3,
                'padding'     : lambda x:1,},
            kwinit={
                'bias'        : False}, )
        self.register(
            'conv5x5', nn.Conv2d,
            kwmap={
                'kernel_size' : lambda x:5,
                'padding'     : lambda x:3,},
            kwinit={
                'bias'        : False}, )
        self.register(
            'conv3x3d', nn.Conv2d,
            kwmap={
                'kernel_size' : lambda x:3,
                'padding'     : lambda x:x['dilation'],},
            kwinit={
                'dilation'    : 1,
                'bias'        : False}, )

        # general bn
        self.register('bn'    , nn.BatchNorm2d)
        self.register('syncbn', nn.SyncBatchNorm)

        # general relu
        self.register('relu'  , nn.ReLU)
        self.register('relu6' , nn.ReLU6)
        self.register(
            'lrelu', nn.LeakyReLU,
            kwargparse={
                0 : ['negative_slope', float]
            }, )

        # general dropout
        self.register(
            'dropout', nn.Dropout,
            kwargparse={
                0 : ['p', float]
            }, )
        self.register(
            'dropout2d', nn.Dropout2d,
            kwargparse={
                0 : ['p', float]
            }, )

    def register(self,
                 cname,
                 objf,
                 kwargparse={},
                 kwmap={},
                 kwinit={},):
        self.component[cname] = [objf, kwargparse, kwmap, kwinit]

    def __call__(self, cname):
        return copy.deepcopy(self.component[cname])

def register(cname,
             kwargparse={},
             kwinit={},
             kwmap={}):
    def wrapper(class_):
        get_component().register(
            cname, class_, kwargparse, kwmap, kwinit)
        return class_
    return wrapper

class nn_component(object):
    def __init__(self,
                 type=None):
        if type is None:
            self.f = None
            return

        type = type.split('|')
        type, para = type[0], type[1:]

        self.f, kwargparse, self.kwmap, self.kwpre = get_component()(type)

        self.kwpost = {}
        for i, parai in enumerate(para):
            fieldname, fieldtype = kwargparse[i]
            self.kwpost[fieldname] = fieldtype(parai)

    def __call__(self, *args, **kwargs):
        """
        The order or priority goes with the following order:
            kwpre -> kwargs(input) -> kwmap -> kwpost
        """
        if self.f is None:
            return identity()
        kw = copy.deepcopy(self.kwpre)
        kw.update(kwargs)
        kwnew = {fn:ff(kw) for fn, ff in self.kwmap.items()}
        kw.update(kwnew)
        kw.update(self.kwpost)
        return self.f(*args, **kw)

def conv_bn_relu(conv_type, bn_type, relu_type):
    return (
        nn_component(conv_type),
        nn_component(bn_type),
        nn_component(relu_type), )

class conv_block(nn.Module):
    """
    Common layers follows the template:
        [conv+bn+relu] x n + conv<+bn><+relu>
    """
    def __init__(self,
                 c_n,
                 para = [3, 1, 1],
                 conv_type='conv',
                 bn_type='bn',
                 relu_type='relu',
                 conv_bias = False,
                 last_conv_bias = True,
                 end_with = 'conv',
                 **kwargs):
        super().__init__()
        self.layer_n = len(c_n)-1
        self.end_with = end_with
        if self.layer_n < 1:
            # need at least the input and output channel number
            raise ValueError
        conv, bn, relu = conv_bn_relu(conv_type, bn_type, relu_type)

        ks, st, pd = para
        # layers except the last
        for i in range(self.layer_n-1):
            ca_n, cb_n = c_n[i], c_n[i+1]
            setattr(
                self, 'conv{}'.format(i),
                conv(ca_n, cb_n, ks, st, pd, bias=conv_bias))
            setattr(
                self, 'bn{}'.format(i),
                bn(cb_n))

        self.relu = relu(inplace=True)

        # last layer
        i = self.layer_n-1
        setattr(
            self, 'conv{}'.format(i),
            conv(c_n[i], c_n[i+1], ks, st, pd, bias=last_conv_bias))
        if end_with == 'conv':
            pass
        elif end_with in ['bn', 'relu']:
            setattr(
                self, 'bn{}'.format(i),
                bn(c_n[i+1]))

    def forward(self, x, debug=False):
        for i in range(self.layer_n-1):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = getattr(self, 'bn{}'.format(i))(x)
            x = self.relu(x)

        i = self.layer_n-1
        x = getattr(self, 'conv{}'.format(i))(x)

        if self.end_with == 'bn':
            x = getattr(self, 'bn{}'.format(i))(x)
        elif self.end_with == 'relu':
            x = getattr(self, 'bn{}'.format(i))(x)
            x = self.relu(x)
        return x

def common_init(m):
    if isinstance(m, (
            nn.Conv2d,
            nn.ConvTranspose2d,)):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (
            nn.BatchNorm2d,
            nn.SyncBatchNorm,)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        try:
            import inplace_abn
            if isinstance(m, (
                    inplace_abn.InPlaceABN,)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        except:
            pass

def init_module(module):
    """
    Args:
        module: [nn.module] list or nn.module
            a list of module to be initialized.
    """
    if isinstance(module, (list, tuple)):
        module = list(module)
    else:
        module = [module]

    for mi in module:
        for mii in mi.modules():
            common_init(mii)
