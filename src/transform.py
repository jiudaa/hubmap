import numpy as np
import albumentations
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return albumentations.Compose(
    [albumentations.HorizontalFlip(p=0.5),
     albumentations.VerticalFlip(p=0.5),
     albumentations.Rotate(limit=180,p=0.7),
#      albumentations.RandomBrightness(limit=0.6,p=0.5),
#      albumentations.Cutout
     albumentations.OneOf([
         albumentations.RandomContrast(),
         albumentations.RandomGamma(),
         albumentations.RandomBrightness(),
     ],p=0.3),
     albumentations.OneOf([
         albumentations.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
         albumentations.GridDistortion(),
         albumentations.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    albumentations.ShiftScaleRotate(p=0.2),
    albumentations.Resize(512,512,always_apply=True),
    ],p=1.)

def get_valid_transforms():
    return albumentations.Compose([
        albumentations.Resize(512,512,always_apply=True),
    ],p=1.)