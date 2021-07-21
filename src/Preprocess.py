import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from tqdm import tqdm
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import zipfile
import pandas as pd


BASE_PATH = '../../data/train'
sz = 512
reduce = 2
MASK = '../../data/train.csv'
DATA = '../../data/train/'
OUT_TRAIN = '../../data/processed/train.zip'
OUT_MASK = '../../data/processed/mask.zip'
os.makedirs('../../data/processed/',exist_ok=True)

df_masks = pd.read_csv(MASK,index_col='id')

def enc2mask(mask_rle,shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width, height) of array to return 
    return: numpy array (width, height)
    '''
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x,dtype=int) for x in (s[0:][::2],s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1],dtype=np.uint8)
    for lo,hi in zip(starts,ends):
        img[lo:hi]=1
    return img.reshape(shape).T

# data_train = pd.read_csv(MASK)
# for i in data_train.id:
#     rle2mask()
# directly read cannot be loaded into 16GB RAM 30000*40000
# use rasterio to load image part by part

s_th = 40 #saturation blank threshold
p_th = 1000*(sz//256)**2  #threshold for the minimum number of pixels
class HuBMAPDataset(Dataset):
    def __init__(self,idx,sz=sz,reduce=reduce,encs=None):
        self.data = rasterio.open(os.path.join(DATA,idx+'.tiff'))
        # issues with format
        if self.data.count!=3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets)>0:
                for i,subdataset in enumerate(subdatasets,0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce*sz
        self.pad0 = (self.sz - self.shape[0]%self.sz)%self.sz
        self.pad1 = (self.sz - self.shape[1]%self.sz)%self.sz
        self.n0max = (self.shape[0]+self.pad0)//self.sz
        self.n1max = (self.shape[1]+self.pad1)//self.sz
        self.mask = enc2mask(encs,(self.shape[1],self.shape[0] if encs is not None else None))
        
    def __len__(self):
        return self.n0max*self.n1max
    
    def __getitem__(self,idx):
        # tiles created with adding padding
        # idx = n0*self.n1max + n1
        n0,n1 = idx//self.n1max, idx%self.n1max
        # x0,y0 are the coordinates of the lower left corner of the tile
        x0,y0 = -self.pad0//2+n0*self.sz, -self.pad1//2+n1*self.sz
        
        p00,p01 = max(0,x0), min(x0+self.sz,self.shape[0])
        p10,p11 = max(0,y0), min(y0+self.sz,self.shape[1])
        
        img = np.zeros((self.sz,self.sz,3),np.uint8)
        mask = np.zeros((self.sz,self.sz),np.uint8)
        if self.data.count == 3:
            img[(p00-x0):(p01-x0),(p10-y0):(p11-y0)]=np.moveaxis(self.data.read([1,2,3],
                                        window=Window.from_slices((p00,p01),(p10,p11))),0,-1)
        else:
            for i,layer in enumerate(self.layers):
                img[(p00-x0):(p01-x0),(p10-y0):(p11-y0),i]=layer.read(1,window = Window.from_slices((p00,p01),(p10,11)))
        if self.mask is not None: mask[(p00-x0):(p01-x0),(p10-y0):(p11-y0)] = self.mask[p00:p01,p10:p11]
        if self.reduce != 1:
            img = cv2.resize(img,(self.sz//reduce,self.sz//reduce),interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask,(self.sz//reduce,self.sz//reduce),interpolation=cv2.INTER_NEAREST)
        
        #drop
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        return img,mask, x0,y0, (-1 if (s>s_th).sum()<=p_th or img.sum()<=p_th else idx)
            

x_tot,x2_tot = [],[]
with zipfile.ZipFile(OUT_TRAIN,'w') as img_out,\
    zipfile.ZipFile(OUT_MASK,'w') as mask_out:
    for index,encs in tqdm(df_masks.iterrows(),total=len(df_masks)):
        ds = HuBMAPDataset(index,encs=encs['encoding'])
        for i in range(len(ds)):
            im,m,x0,y0,idx = ds[i]
            if idx<0: continue
            im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
            os.makedirs(f'../../data/processed/train/{index}',exist_ok=True)
            cv2.imwrite(f'../../data/processed/train/{index}/{index}_{x0:05d}_{y0:05d}.png',im)
            cv2.imwrite(f'../../data/processed/train/{index}/{index}_{x0:05d}_{y0:05d}_mask.png',m)

            x_tot.append((im/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((im/255.0)**2).reshape(-1,3).mean(0))
            
            im = cv2.imencode('.png',im)[1]
            img_out.writestr(f'{index}_{x0:05d}_{y0:05d}.png',im)
            m = cv2.imencode('.png',m)[1]
            mask_out.writestr(f'{index}_{x0:05d}_{y0:05d}.png',m)
                
# #image state
# img_avr = np.array(x_tot).mean(0)
# img_std = np.sqrt(np.array(x2_tot).mean(0)-img_avr**2)
