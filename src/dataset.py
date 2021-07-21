from torch.utils.data import Dataset
import cv2
import numpy as np
class HUBMAPDataset(Dataset):
    def __init__(self,ids,transform=None):
        self.ids = ids
        self.transform = transform
        
    def __len__(self):
        return len(self.ids)
    def __getitem__(self,idx):
        name = self.ids[idx]
        image_name = name.split('_')[0]
        img = cv2.imread(f'../../data/processed/train/{image_name}/{name}.png').astype('float32')
        img /= 255.
        mask = cv2.imread(f'../../data/processed/train/{image_name}/{name}_mask.png',0)
        label = (mask.sum()>0)*1
        if self.transform is None:
            img = img.transpose(2,0,1).astype('float32')
            mask = mask.transpose(2,0,1).astype('float32')
            return {'image': img,
                    'mask': mask,
                   'label':label}
        else:

            img = self.transform()(image=img,mask=mask)["image"]
            mask = self.transform()(image=img,mask=mask[:,:,np.newaxis])["mask"]
            
            img = img.transpose(2,0,1).astype('float32')
            mask = mask.transpose(2,0,1).astype('float32')
            
            return {'image': img,
                  'mask': mask,
                  'label':label}