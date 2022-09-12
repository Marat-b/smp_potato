from PIL import Image
from torch.utils.data import Dataset as BaseDataset
import cv2
import os
import numpy as np

class PotatoDataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    # CLASSES = ['unlabelled', 'sky', 'building', 'pole', 'road', 'pavement',
    #            'tree', 'signsymbol', 'fence', 'car',
    #            'pedestrian', 'bicyclist']

    def __init__(
            self,
            images_dir,
            masks_dir,
            # classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        # print(f'self.ids={self.ids}')
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('jpg', 'png')) for image_id in self.ids]

        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        # self.class_values = [cls for cls in range(classes)]
        # print(f'self.class_values={self.class_values}')

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        # image = cv2.imread(self.images_fps[i])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.masks_fps[i], 0)
        image = Image.open(self.images_fps[i])
        mask = Image.open(self.masks_fps[i])
        image = np.array(image)
        mask = np.array(mask)[:, :, 0]
        # print(f'getitem mask={mask}')

        # extract certain classes from mask (e.g. cars)
        # masks = [v for v in self.class_values if np.max(v)>0]
        # mask = np.stack(masks, axis=-1).astype('float')
        # print(f'getitem mask.shape={mask.shape}')
        # print(f'max(mask)={np.max(mask)}')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)