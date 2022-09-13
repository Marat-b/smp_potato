import datetime
import os
import shutil
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
import albumentations as albu

from datas.potato_dataset import PotatoDataset


class Trainer:
    def __init__(self):
       self.classes = 8
       self.activation = 'softmax'
       self.encoder = 'resnet50'
       self.encoder_weights = 'imagenet'

    def get_training_augmentation(self):
        train_transform = [

            albu.HorizontalFlip(p=0.5),

            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
            albu.RandomCrop(height=320, width=320, always_apply=True),

            albu.IAAAdditiveGaussianNoise(p=0.2),
            albu.IAAPerspective(p=0.5),

            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.IAASharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.RandomContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)

    def get_validation_augmentation(self):
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.PadIfNeeded(384, 480)
        ]
        return albu.Compose(test_transform)

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self, preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return albu.Compose(_transform)

    def get_ymd(self):
        now = datetime.datetime.now()
        year = now.year
        month = str(now.month)
        day = str(now.day)
        hour = str(now.hour)
        if len(month) != 2:
            month = '0' + month
        if len(day) != 2:
            day = '0' + day
        if len(hour) != 2:
            hour = '0' + hour
        return '{}{}{}{}'.format(year, month, day, hour)

    def prepare(self):
        train_dataset = PotatoDataset(
            self.train_images_path,
            self.train_masks_path,
            augmentation=self.get_training_augmentation(),
            preprocessing=self.get_preprocessing(self.preprocessing_fn)
        )

        valid_dataset = PotatoDataset(
            self.validate_images_path,
            self.validate_masks_path,
            augmentation=self.get_validation_augmentation(),
            preprocessing=self.get_preprocessing(self.preprocessing_fn)
        )

        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
        # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

        loss = smp.utils.losses.DiceLoss()

        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

        self.optimizer = torch.optim.Adam(
            [
                dict(params=self.model.parameters(), lr=0.0001),
            ]
        )
        # create epoch runners
        # it is a simple loop of iterating over dataloader`s samples
        self.train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=loss,
            metrics=metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True,
        )

        self.valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=loss,
            metrics=metrics,
            device=self.device,
            verbose=True,
        )

    @property
    def preprocessing_fn(self):
        return smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)


    def train(self):
        max_score = 0
        for i in range(0, self.epoch):

            print('\nEpoch: {}'.format(i))
            train_logs = self.train_epoch.run(self.train_loader)
            valid_logs = self.valid_epoch.run(self.valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                # torch.save(model, best_model_file)
                torch.save(self.model, './best_model.pth')
                print('Model saved!')
                shutil.copy(
                    './best_model.pth',
                    os.path.join(self.output_folder, 'dl_potato_model_best{}.pth'.format(self.get_ymd()))
                )

            if i == 25:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

    def main(self, args):
        self.output_folder = args.output_folder
        self.train_masks_path = args.train_masks_path
        self.train_images_path = args.train_images_path
        self.validate_masks_path = args.validate_masks_path
        self.validate_images_path = args.validate_images_path
        self.epoch = args.epoch
        # resume = args.resume
        # ENCODER = 'resnet50'
        # ENCODER_WEIGHTS = 'imagenet'
        # ACTIVATION = 'softmax'
        self.device = args.device
        self.model = smp.DeepLabV3(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            classes=self.classes,
            activation=self.activation,
        )
        # self.preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        # if resume:
        #     max_iter = input('Input max iteration (max_iter):')
        #     lr = input('Input learning rate (lr):')
        #     self._load_cfg_resuming(max_iter=int(max_iter), lr=float(lr))
        # else:
        #     self._load_cfg()
        # self.train_model(output_folder=self.output_folder, resume=resume)
        # self._save_torchscript()
        self.prepare()
        self.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Trainer Composition")
    parser.add_argument(
        "-tm", "--train_masks_dir",
        type=str,
        dest="train_masks_path",
        required=True,
        help="Path of json file of COCO format"
    )
    parser.add_argument(
        "-ti", "--train_images_dir",
        type=str,
        dest="train_images_path",
        required=True,
        help=""
    )
    parser.add_argument(
        "-vm",  "--validate_masks_dir",
        type=str,
        dest="validate_masks_path",
        required=True,
        help=""
    )
    parser.add_argument(
        "-vi",  "--validate_images_dir",
        type=str,
        dest="validate_images_path",
        required=True,
        help=""
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        dest="output_folder",
        required=False,
        default="./",
        help=""
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        dest="device",
        required=False,
        default="cuda",
        help="Device, cuda or cpu, default - cuda"
    )
    parser.add_argument(
        "-e", "--epoch",
        type=int,
        dest="epoch",
        required=False,
        default=1,
        help="Number of epochs"
    )
    # parser.add_argument(
    #     "--resume",
    #     type=bool,
    #     dest="resume",
    #     default=False,
    #     required=False,
    #     help=""
    # )
    args = parser.parse_args()
    pt = Trainer()
    pt.main(args)

