"""
(C) MK & ASL & AL

Functions to work with DICOMs.


Started: 2019.11.21
Modified: 2019.11.21
"""

import os
import numpy as np
import pydicom as dicom


def convertDICOM2NPY(savePth='', PRINT=False):
    """
    Reads all *.dcm files in a current folder, stores them in a 3D np.ndarray and save a matrix to one *.npy file.
    
    Based on an example from pydicom.github.io:
    https://pydicom.github.io/pydicom/stable/auto_examples/image_processing/reslice.html#sphx-glr-auto-examples-image-processing-reslice-py
    
    Started: 2019.11.11
    Modified: 2019.11.21
    """
    
    # find all *dcm files    
    files = os.listdir('.')
    files = [f for f in files if f.endswith('.dcm')]
    files.sort()
   
    # load all DICOM files
    dcms = []
    for f in files:
        dcms.append(dicom.dcmread(f))
    if PRINT:
        print("\nfile count in dcms: {}".format(len(dcms)))
    
    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for d in dcms:
        if hasattr(d, 'SliceLocation'):
            slices.append(d)
        else:
            skipcount = skipcount + 1
    if PRINT:
        print("skipped files without 'SliceLocation': {}".format(skipcount))
    
    # ensure they are in the correct order - sort them
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    
    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    pixelType = slices[0].pixel_array.dtype
    img3d = np.zeros(img_shape, dtype=pixelType)
    
    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d
    
    # if saveName is provided images are saved to *.npy file in the DCM2NII folder
    if len(savePth):        
        np.save(savePth, img3d)
        if PRINT:
            print('saved to: %s' % savePth)
    
    return img3d


def getStatistics(im3d, savePthSL='', savePthIm3d='', PRINT=False):
    """
    Calculates min, max, mean, std values for each 2D DCIOM slice
    and for a 3D array (that contains all 2D slices).
    
    input: 3D array that contains all 2D slices
    
    output: 
        statsSL - matrix with ~170 rows (for each 2D slice) and 4 columns (min/max/mean/std)
        statsIm3D - matrix of 4 elements (columns). These are global values the whole 3D array (min/max/mean/std)
        
        Both arrays are written to separate folders: savePthSL and savePthIm3d
    """
    # statistisc for each slide separately
    sz = im3d.shape[-1] # number of slices
    
    statsSL = np.zeros((sz, 4), dtype=np.float32)
    for i in range(sz):
        statsSL[i] = im3d[:,:,i].min(), im3d[:,:,i].max(), im3d[:,:,i].mean(), im3d[:,:,i].std()

    # statistics for a 3D image
    statsIm3d = np.array([im3d.min(), im3d.max(), im3d.mean(), im3d.std()], dtype=np.float32)

    # save to statistics to two separate *.npy files
    if len(savePthSL): # STATSL folder
        np.save(savePthSL, statsSL)
        if PRINT:
            print('saved to %s' % savePthSL)
    if len(savePthIm3d): # STATIM3D folder
        np.save(savePthIm3d, statsIm3d)
        if PRINT:
            print('saved to: %s' % savePthIm3d)