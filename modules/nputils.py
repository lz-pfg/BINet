import os
import os.path as osp
import numpy as np
import numpy.random as npr
import cv2
import copy
import collections

def batch_op(dim):
    def unsqueeze_dim0(x):
        if x is None:
            return None
        return np.expand_dims(x, axis=0)

    def squeeze_dim0(x):
        if x is None:
            return None
        return x[0]

    def reshape_dimX(x, d, s):
        if x is None:
            return None
        s = copy.deepcopy(s)
        s.extend(list(x.shape[d:]))
        return np.reshape(x, s)

    def wrapper(func):
        def inner(self, x, *argv, **kwargs):
            x = copy.deepcopy(x)
            argv = copy.deepcopy(argv)
            kwargs = copy.deepcopy(kwargs)

            cond = [
                isinstance(x, np.ndarray)]
            cond += [
                (arg is None) or isinstance(arg, np.ndarray) \
                for arg in argv]
            cond += [
                (arg is None) or isinstance(arg, np.ndarray) \
                for _, arg in kwargs.items()]

            if sum(cond) != len(cond):
                raise ValueError

            if len(x.shape) < dim:
                raise ValueError

            if len(x.shape) == dim:
                x = unsqueeze_dim0(x)
                argv = [
                    unsqueeze_dim0(arg) \
                    for arg in argv]
                kwargs = {
                    k: unsqueeze_dim0(arg) \
                    for k, arg in kwargs.items()}
                r = func(self, x, *argv, **kwargs)
                if isinstance(r, tuple):
                    rn = [
                        squeeze_dim0(ri) \
                        for ri in r]
                    return tuple(rn)
                else:
                    return squeeze_dim0(r)

            if len(x.shape) == dim + 1:
                n = x.shape[0]
                cond = [
                    (arg is None) or (arg.shape[0] == n) \
                    for arg in argv]
                cond += [
                    (arg is None) or (arg.shape[0] == n) \
                    for _, arg in kwargs.items()]
                if sum(cond) != len(cond):
                    raise ValueError
                return func(self, x, *argv, **kwargs)

            if len(x.shape) > dim + 1:
                d = len(x.shape) - dim
                n = list(x.shape[0:d])
                n_new = [np.prod(n)]

                cond = [
                    (arg is None) or list(arg.shape[0:d]) == n \
                    for arg in argv]
                cond += [
                    (arg is None) or list(arg.shape[0:d]) == n \
                    for _, arg in kwargs.items()]
                if sum(cond) != len(cond):
                    raise ValueError

                x = reshape_dimX(x, d, n_new)
                argv = [
                    reshape_dimX(arg, d, n_new) \
                    for arg in argv]
                kwargs = {
                    k: reshape_dimX(arg, d, n_new)
                    for k, arg in kwargs.items()}

                r = func(self, x, *argv, **kwargs)
                if isinstance(r, tuple):
                    rn = [
                        reshape_dimX(ri, 1, n) \
                        for ri in r]
                    return tuple(rn)
                else:
                    return reshape_dimX(r, 1, n)

        return inner

    return wrapper


def batchwise_mean(x):
    return np.mean(np.reshape(x, (x.shape[0], -1)), axis=1, keepdims=1)


def batchwise_sum(x):
    return np.sum(np.reshape(x, (x.shape[0], -1)), axis=1, keepdims=1)

class nearest_interpolate_2d(object):
    def __init__(self,
                 size,
                 align_corners=True):
        self.size = size
        self.align_corners = align_corners
        self.eps = np.finfo(np.float64).eps

    @batch_op(2)
    def __call__(self,
                 x):
        _, h, w = x.shape
        if self.align_corners:
            hn = np.linspace(0, h - 1, num=self.size[0])
            wn = np.linspace(0, w - 1, num=self.size[1])
        else:
            hn = np.linspace(0, h, num=self.size[0] + 1)[:-1]
            wn = np.linspace(0, w, num=self.size[1] + 1)[:-1]
            hn = hn + h / (self.size[0] * 2) - 0.5
            wn = wn + w / (self.size[1] * 2) - 0.5

        hnsp = [hn[i] + self.eps * hn[i] for i in range(0, len(hn) // 2)] \
               + [hn[i] - self.eps * hn[i] for i in range(len(hn) // 2, len(hn))]
        wnsp = [wn[i] + self.eps * wn[i] for i in range(0, len(wn) // 2)] \
               + [wn[i] - self.eps * wn[i] for i in range(len(wn) // 2, len(wn))]
        hn = np.rint(hnsp).astype(int)
        wn = np.rint(wnsp).astype(int)
        y = x[:, hn, :]
        y = y[:, :, wn]
        return y.copy()


class interpolate_2d(object):
    def __init__(self,
                 size,
                 mode='nearest',
                 align_corners=True,
                 **kwargs):
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    @batch_op(2)
    def __call__(self,
                 x):
        if self.mode == 'nearest':
            y = nearest_interpolate_2d(self.size, align_corners=self.align_corners)(x)
        else:
            import torch
            x = torch.as_tensor(x).unsqueeze(1).float()
            y = torch.nn.functional.interpolate(
                x, self.size, mode=self.mode,
                align_corners=self.align_corners)
            y = y.squeeze(1).numpy()
        return y


class auto_interpolate_2d(object):
    def __init__(self,
                 size,
                 align_corners=True,
                 **kwargs):
        self.size = size
        self.align_corners = align_corners
        self.int_types = [np.uint8, np.int64]
        self.float_types = [np.float32]

    @batch_op(2)
    def __call__(self,
                 x):
        dtype = x.dtype
        if dtype in self.int_types:
            y = nearest_interpolate_2d(self.size, align_corners=self.align_corners)(x)
        elif dtype in self.float_types:
            import torch
            x = torch.as_tensor(x).unsqueeze(1).float()
            y = torch.nn.functional.interpolate(
                x, self.size, mode='bilinear',
                align_corners=self.align_corners)
            y = y.squeeze(1).numpy()
        return y


class normalize_2d(object):
    def __init__(self,
                 mean,
                 std,
                 **kwargs):
        self.mean = np.array(mean)
        self.std = np.array(std)

    @batch_op(3)
    def __call__(self,
                 x):
        y = x - self.mean[:, np.newaxis, np.newaxis]
        y /= self.std[:, np.newaxis, np.newaxis]
        return y
