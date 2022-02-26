#!/usr/bin/python3
#coding=utf-8

import os
import shutil
import numpy as np
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
import warnings
from glob import glob
import concurrent.futures
from config_training import config

def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM!=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing, isflip

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask

def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg*255).astype('uint8')
    return newimg

def savenpy_luna(id, annos, filelist, luna_segment, luna_data, savepath):
    """
    Note: Dr. Chen adds malignancy label, so the label becomes (z,y,x,d,malignancy), <- but I cancelled it !
    """
    islabel = True
    isClean = True
    resolution = np.array([1, 1, 1])
    name = filelist[id]

    # Load mask, and calculate extendbox from the mask
    Mask, origin, spacing, isflip = load_itk_image(os.path.join(luna_segment, name+'.mhd'))
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T


    if isClean:
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        extramask = dilatedMask ^ Mask  # '-' substration is deprecated in numpy, use '^'
        bone_thresh = 210
        pad_value = 170

        sliceim, origin, spacing, isflip = load_itk_image(os.path.join(luna_data, name+'.mhd'))
        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('{}: flip!'.format(name))
        sliceim = lumTrans(sliceim)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                            extendbox[1,0]:extendbox[1,1],
                            extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]

        np.save(os.path.join(savepath, name + '_clean.npy'),sliceim)

        np.save(os.path.join(savepath, name+'_spacing.npy'), spacing)
        np.save(os.path.join(savepath, name+'_extendbox.npy'), extendbox)
        np.save(os.path.join(savepath, name+'_origin.npy'), origin)
        np.save(os.path.join(savepath, name+'_mask.npy'), Mask)


    if islabel:
        this_annos = np.copy(annos[annos[:,0] == name])
        label = []

        if len(this_annos)>0:
            for c in this_annos:   # unit in mm  -->  voxel
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)  # (z,y,x)
                if isflip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]   # flip in y and x coordinates
                d = c[4]/spacing[1]
                try:
                    malignancy = int(c[5])
                except IndexError:
                    malignancy = 0
                # label.append(np.concatenate([pos,[d],[malignancy]]))  # (z,y,x,d,malignancy)
                label.append(np.concatenate([pos,[d]]))  # (z,y,x,d)
            
        label = np.array(label)

        # Voxel --> resample to (1mm,1mm,1mm) voxel coordinate
        if len(label)==0:
            # label2 = np.array([[0,0,0,0,0]])
            label2 = np.array([[0,0,0,0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            label2[3] = label2[3]*spacing[1]/resolution[1]
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            # label2 = label2[:5].T   #(z,y,x,d,malignancy)
            label2 = label2[:4].T   #(z,y,x,d)

        np.save(os.path.join(savepath, name+'_label.npy'), label2)
        
    print('{} is done.'.format(name))

def preprocess_luna():
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocess_luna'

    print('starting preprocessing luna')
    
    if True:
        exist_files = {f.split('_clean.npy')[0] for f in os.listdir(savepath) if f.endswith('_clean.npy')}
        filelist = {f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd')}
        filelist = list(filelist - exist_files)
        annos = np.array(pandas.read_csv(luna_label))

        if not os.path.isdir(savepath):
            os.mkdir(savepath)

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(savenpy_luna, f, annos=annos, filelist=filelist,
                                       luna_segment=luna_segment, luna_data=luna_data, savepath=savepath):f for f in range(len(filelist))}
            for future in concurrent.futures.as_completed(futures):
                filename = filelist[futures[future]]
                try:
                    _ = future.result()
                except:
                    print('{} failed.'.format(filename))


    print('end preprocessing luna')
    f = open(finished_flag,"w+")
    f.close()
    return
    

if __name__=='__main__':    
    # Pre-process LUNA16 MHD files
    preprocess_luna()
