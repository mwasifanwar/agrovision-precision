import albumentations as A
import numpy as np
import cv2

class DataAugmentation:
    def __init__(self):
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
        ])
    
    def augment_image(self, image, mask=None):
        if mask is not None:
            augmented = self.augmentation_pipeline(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.augmentation_pipeline(image=image)
            return augmented['image'], None
    
    def augment_batch(self, images, masks=None):
        augmented_images = []
        augmented_masks = [] if masks is not None else None
        
        for i, image in enumerate(images):
            if masks is not None:
                aug_img, aug_mask = self.augment_image(image, masks[i])
                augmented_images.append(aug_img)
                augmented_masks.append(aug_mask)
            else:
                aug_img, _ = self.augment_image(image)
                augmented_images.append(aug_img)
        
        if masks is not None:
            return np.array(augmented_images), np.array(augmented_masks)
        else:
            return np.array(augmented_images)