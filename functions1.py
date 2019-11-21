"""
(C) MK & ASL & AL

Additional functions to ADNI


Started: 2019.11.21
Modified: 2019.11.21
"""

import os


def list_2_txt(flist, savePath):
    with open(savePath, 'w') as f:
        for item in flist:
            f.write("%s\n" % item)
    print('list saved to: %s' % savePath)
    
    
def info(img, name='image'):
    print(' {}: min={:.2f}, aver={:.2f},  max={:.2f}, \
shape={},dtype={}'.format(name, img.min(), img.mean(), img.max(), img.shape, img.dtype))
    
    
def text_wrap(fname = ''):
    """
    C: 2019.06.18
    M: 2019.06.22
    """
    if not len(fname):
        fname=os.path.basename(__file__)
    print()
    p = len(fname) + 8
    print(p * '#')
    print("### %s ###" % fname)
    print(p * '#')
    
def pthInfo():
    print("Current folder is:\n\t%s"  % os.getcwd())
   

def progress(curK, allK):
    print("Progress: {}/{}".format(curK, allK ), end='\r')