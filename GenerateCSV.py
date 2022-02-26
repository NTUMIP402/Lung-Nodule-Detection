import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import json
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import math
import SimpleITK as sitk
import csv
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--cross', default=None, type=str, metavar='N',
                    help='which data cross be used')
parser.add_argument('--epoch', default=None, type=str, metavar='N',
                    help='which data cross be used')

args = parser.parse_args()



def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]

        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                break
        else:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing, isflip

def main(bbox_path, preprocess_path, lunaseg_path, save_file):
    total_list = []
    epochs = args.epoch
    epochs = epochs.split('.') 
    count = 0
    for i in range(5):
        total_list.append([])        
        with Path('test_0222_%s/LUNA_test.json' %str(i+1)).open('rt', encoding='utf-8') as fp:
            idcs = json.load(fp)
        for x in tqdm(range(len(idcs))):            
            pbb = np.load('%s%s/bbox_%s/%s_pbb.npy' %(bbox_path, str(i+1), epochs[i], idcs[x]), mmap_mode='r')            
            lbb = np.load("%s%s_label.npy" % (preprocess_path, idcs[x]), allow_pickle=True)            
            pbb = nms(pbb, 0.1)            

            Mask,origin,spacing,isflip = load_itk_image('%s%s.mhd' %(lunaseg_path, idcs[x]))
            
            origin = np.load('%s%s_origin.npy' %(preprocess_path, idcs[x]), mmap_mode='r')
            spacing = np.load('%s%s_spacing.npy' %(preprocess_path, idcs[x]), mmap_mode='r')
            resolution = np.array([1, 1, 1])
            extendbox = np.load('%s%s_extendbox.npy' %(preprocess_path, idcs[x]), mmap_mode='r')
                        
            pbb = np.array(pbb[:, :-1])            
            pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:,0], 1).T)
            pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)

            if isflip:
                Mask = np.load('%s%s_mask.npy' %(preprocess_path, idcs[x]), mmap_mode='r')
                pbb[:, 2] = pbb[:, 2] - Mask.shape[1]
                pbb[:, 3] = pbb[:, 3] - Mask.shape[2] 
                
            pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)            

            rowlist = []
            for nk in range(pos.shape[0]):                 
                rowlist.append([idcs[x], pos[nk, 2], pos[nk, 1], pos[nk, 0], pbb[nk,0]])
            
            total_list[i].append(rowlist)

        with open(save_file, "w") as f:
            first_row = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
            
            f.write("%s,%s,%s,%s,%s\n" %(first_row[0], first_row[1], first_row[2], first_row[3], first_row[4]))
            
            for k in total_list:
                for i in k:
                    for j in i:
                        f.write("%s,%.9f,%.9f,%.9f,%.9f\n" %(j[0], j[1], j[2], j[3], j[4]))
        
if __name__=='__main__':
    main(bbox_path='./results/'+args.model+'_testcross', preprocess_path='../data/preprocess/all/', lunaseg_path='../data/LUNA16/seg-lungs-LUNA16/', save_file=args.model+'_80_all.csv')




    