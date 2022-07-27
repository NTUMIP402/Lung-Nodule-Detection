#!/usr/bin/python3
#coding=utf-8
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
import warnings
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
import json
from pathlib import Path



class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, split, config, phase='train', split_comber=None):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.max_stride = config['max_stride']       
        self.stride = config['stride']       
        sizelim = config['sizelim']/config['reso']
        sizelim2 = config['sizelim2']/config['reso']
        sizelim3 = config['sizelim3']/config['reso']
        sizelim4 = config['sizelim4']/config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']  # random ratio for sample augmentation == 0.3
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        if isinstance(split, list):
            idcs = split
        elif isinstance(split, str) and Path(split).suffix == '.json':
            with Path(split).open('rt', encoding='utf-8') as fp:
                idcs = json.load(fp)
        # A fix for python2 ascii compatibility
        idcs = [str(i, encoding='utf-8') if not isinstance(i, str) else i for i in idcs]

        if phase!='test':
            idcs = [f for f in idcs if f not in self.blacklist]

        self.channel = config['channel']  # channel==1
        self.filenames = [os.path.join(data_dir, '{}_clean.npy'.format(idx)) for idx in idcs]
        
        labels = []
        
        for idx in idcs:
            l = np.load(Path(data_dir)/'{}_label.npy'.format(idx), allow_pickle=True) # l = [z, y, x, d]
            if np.all(l==0):
                l = np.array([])
            labels.append(l)
        self.sample_bboxes = labels

        # Balance nodules of different diameters by sizelim. sizelim2, sizelim3 by augment bigger nodules, which are fewer in dataset.
        if self.phase != 'test':
            self.bboxes = []
            for index, label in enumerate(labels):
                if len(label) > 0 :
                    for t in label:   # t = (z, y, x, d)
                        if t[3] > sizelim:
                            self.bboxes.append([np.concatenate([[index], t])])
                        if t[3] > sizelim2:
                            self.bboxes += [[np.concatenate([[index], t])]] * 2
                        if t[3] > sizelim3:
                            self.bboxes += [[np.concatenate([[index], t])]] * 4
                        if t[3] > sizelim4:
                            self.bboxes += [[np.concatenate([[index], t])]] * 8

            # Finally, a balanced label collection is generated.
            if len(self.bboxes) > 0:
                self.bboxes = np.concatenate(self.bboxes, axis=0)
            else:
                self.bboxes = np.array(self.bboxes)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))

        isRandomImg  = False
        if self.phase == 'train' or self.phase == 'val':
            if idx >= len(self.bboxes):
                isRandom = True
                idx = np.random.randint(0, len(self.bboxes))
                isRandomImg = False
            else:
                isRandom = False
        elif self.phase == 'test':
            isRandom = False
        
        if self.phase == 'train' or self.phase == 'val':
            if not isRandomImg:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = np.load(filename)[0:self.channel]
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale=isScale, isRand=isRandom)

                if self.phase=='train' and not isRandom:
                     sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                        ifflip=self.augtype['flip'], ifrotate=self.augtype['rotate'], ifswap=self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                imgs = np.load(filename)[0:self.channel]
                bboxes = self.sample_bboxes[randimid]
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)

            try:
                label = self.label_mapping(sample.shape[1:], target, bboxes, filename)
            except ZeroDivisionError:
                raise Exception('Bug in {}'.format(os.path.basename(filename).split('_clean')[0]))

            sample = (sample.astype(np.float32)-128)/128

            # print('sample_shape: ', sample.shape, '  label_shape: ', label.shape)
            return torch.from_numpy(sample), torch.from_numpy(label), coord


        elif self.phase == 'test':
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0,0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant', constant_values=self.pad_value)
            xx,yy,zz = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]//self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[2]//self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[3]//self.stride), indexing='ij')
            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
            imgs, nzhw = self.split_comber.split(imgs)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=self.split_comber.side_len//self.stride,
                                                    max_stride=self.split_comber.max_stride//self.stride,
                                                    margin=self.split_comber.margin//self.stride)
            assert np.all(nzhw==nzhw2)
            imgs = (imgs.astype(np.float32)-128)/128
            return torch.from_numpy(imgs.astype(np.float32)), bboxes, torch.from_numpy(coord2.astype(np.float32)), np.array(nzhw)

    def __len__(self):
        if self.phase == 'train':
            return int(len(self.bboxes)//(1-self.r_rand))
        elif self.phase =='val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)


class NoduleMalignancyDetector(Dataset):
    """ Save malignancy label of each nodule in label.npy with [z, y, x, d, malignancy]

    """
    def __init__(self, data_dir, split, config, phase='train', split_comber=None):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.max_stride = config['max_stride']
        self.stride = config['stride']
        sizelim = config['sizelim'] / config['reso']
        sizelim2 = config['sizelim2'] / config['reso']
        sizelim3 = config['sizelim3'] / config['reso']
        sizelim4 = config['sizelim4'] / config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']  # random ratio for sample augmentation == 0.3
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        if isinstance(split, list):
            idcs = split
        elif isinstance(split, str) and Path(split).suffix == '.json':
            with Path(split).open('rt', encoding='utf-8') as fp:
                idcs = json.load(fp)
        # A fix for python2 ascii compatibility
        idcs = [str(i, encoding='utf-8') if not isinstance(i, str) else i for i in idcs]

        if phase != 'test':
            idcs = [f for f in idcs if f not in self.blacklist]

        self.channel = config['channel']  # channel==1
        self.filenames = [os.path.join(data_dir, '{}_clean.npy'.format(idx)) for idx in idcs]

        self.sample_bboxes = []

        for idx in idcs:
            l = np.load(Path(data_dir)/'{}_label.npy'.format(idx))   # l = [z, y, x, d, malignancy]
            if np.all(l==0):
                l = np.array([])
            self.sample_bboxes.append(l)


        # Balance nodules of different diameters by sizelim. sizelim2, sizelim3 by augment bigger nodules, which are fewer in dataset.
        if self.phase != 'test':
            self.bboxes = []
            for index, label in enumerate(self.sample_bboxes):
                if len(label) > 0:
                    for t in label:  # t = (z, y, x, d, malignancy)
                        if t[3] > sizelim:
                            self.bboxes.append([np.concatenate([[index], t])])
                        if t[3] > sizelim2:
                            self.bboxes += [[np.concatenate([[index], t])]] * 2
                        if t[3] > sizelim3:
                            self.bboxes += [[np.concatenate([[index], t])]] * 4
                        if t[3] > sizelim4:
                            self.bboxes += [[np.concatenate([[index], t])]] * 8

            # Finally, a balanced label collection is generated.
            if len(self.bboxes) > 0:
                self.bboxes = np.concatenate(self.bboxes, axis=0)
            else:
                self.bboxes = np.array(self.bboxes)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))

        isRandomImg = False
        if self.phase == 'train' or self.phase == 'val':
            if idx >= len(self.bboxes):
                isRandom = True
                idx = idx % len(self.bboxes)
                isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        elif self.phase == 'test':
            isRandom = False

        if self.phase == 'train' or self.phase == 'val':
            if not isRandomImg:
                bbox = self.bboxes[idx]   # bbox = (idx, z, y, x, d, malignancy)
                filename = self.filenames[int(bbox[0])]
                imgs = np.load(filename)[0:self.channel]
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.phase == 'train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:5], bboxes, isScale=isScale, isRand=isRandom)
                try:
                    malignancy = bbox[5]
                except IndexError:
                    malignancy = 0

                if self.phase == 'train' and not isRandom:
                    sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                            ifflip=self.augtype['flip'],
                                                            ifrotate=self.augtype['rotate'],
                                                            ifswap=self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                imgs = np.load(filename)[0:self.channel]
                bboxes = self.sample_bboxes[randimid]
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)
                malignancy = 0  # it's randomly selected, so the malignancy is unknown.

            try:
                label = self.label_mapping(sample.shape[1:], target, bboxes, filename)
            except ZeroDivisionError:
                raise Exception('Bug in {}'.format(os.path.basename(filename).split('_clean')[0]))

            sample = (sample.astype(np.float32) - 128) / 128
            return torch.from_numpy(sample), torch.from_numpy(label), coord, torch.tensor(malignancy, dtype=torch.int)

        elif self.phase == 'test':
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                          constant_values=self.pad_value)
            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] // self.stride), indexing='ij')
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
            imgs, nzhw = self.split_comber.split(imgs)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=self.split_comber.side_len // self.stride,
                                                    max_stride=self.split_comber.max_stride // self.stride,
                                                    margin=self.split_comber.margin // self.stride)
            assert np.all(nzhw == nzhw2)
            imgs = (imgs.astype(np.float32) - 128) / 128
            return torch.from_numpy(imgs.astype(np.float32)), bboxes, torch.from_numpy(coord2.astype(np.float32)), np.array(nzhw)

    def __len__(self):
        if self.phase == 'train':
            return int(len(self.bboxes) // (1 - self.r_rand))
        elif self.phase == 'val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)


class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']  #int: [96,96,96]
        self.bound_size = config['bound_size']  #12
        self.stride = config['stride']  #4
        self.pad_value = config['pad_value']  #170

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        if isScale:
            # target: (z,y,x,d)
            radiusLim = [8., 120.]
            scaleLim = [0.75, 1.25]
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1]),
                          np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            else:
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
                e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])

            if s > e:
                start.append(np.random.randint(e, s))  # !
            else:
                start.append(int(target[i] - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2)))

        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] // self.stride),
                                 np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] // self.stride),
                                 np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] // self.stride),
                                 indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        pad = []
        pad.append([0, 0])

        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])

        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)

        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=2)
            newpad = self.crop_size[0] - crop.shape[1:][0]

            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)

            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale
        return crop, target, bboxes, coord


class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])  #4
        self.num_neg = int(config['num_neg'])  #800
        self.th_neg = config['th_neg']   #0.02
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']   #0.5
        elif phase == 'val':
            self.th_pos = config['th_pos_val']     #1

    def __call__(self, input_size, target, bboxes, filename):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos

        output_size = []
        for i in range(3):
            assert(input_size[i] % stride == 0), 'input_size[{}]={}, stride={}, filename={}'.format(i, input_size[i], str(stride), filename)
            output_size.append(input_size[i] // stride)

        # Initialize all grid labels to -1
        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)  #(24, 24, 24, #anchor, 5)
        offset = ((stride.astype('float')) - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        # Find the positively-labeled grids in bboxes, and set them to 0
        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 0] = 0

        if self.phase == 'train' and self.num_neg > 0:
            # Now, all grids which are labeled as -1 are negative grids.
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)

            # Select num_neg(=800) of them, set as -1, leave all others(including positive grid) to 0
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1

        # If no target in this crop, return negative grids(labeled as -1) only.
        if np.isnan(target[0]):
            return label

        # Locate the target on the grids
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)

        if len(iz) == 0:
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
        else:  # randomly choose one if there is more than one positive grid
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]

        # Calculate the difference ratio of (z,h,w,d) between target and positive grid(=pos) relative to anchor
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        return label


def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = (np.random.rand()-0.5)*20
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                coord = rotate(coord,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
            else:
                counter += 1
                if counter ==3:
                    break
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]
            
    if ifflip:
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
        for ax in range(3):
            if flipid[ax]==-1:
                target[ax] = np.array(sample.shape[ax+1])-target[ax]
                bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]

    # normal = np.random.uniform(-0.1, 0.1, sample.shape)
    # if np.random.randint(0, 2):
    #     sample = sample + normal

    return sample, target, bboxes, coord 

def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox

    if d == 0:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap

    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]
        
        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]
            
        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
        
        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))

        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis=1)
        
        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0
        
        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))
        
        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))
        
        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union
        mask = iou >= th

        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]

        return iz, ih, iw

def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

