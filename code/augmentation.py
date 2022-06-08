import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(args):

    return A.Compose([
                     A.Resize(always_apply=False, p=1.0, height=384, width=384, interpolation=0),
                     A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
                     A.OneOf([
                     # A.CLAHE(always_apply=False, p=1.0, clip_limit=(1, 4), tile_grid_size=(8, 8)),
                     A.ChannelShuffle(always_apply=False, p=1.0),
                     A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.20), brightness_by_max=True),
                     A.ToGray(always_apply=False, p=1.0),
                     A.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-20, 20), sat_shift_limit=(-40, 40), val_shift_limit=(-20, 20)),
                        ],p=0.3),
                     A.Cutout(always_apply=False, p=0.3, num_holes=4, max_h_size=17, max_w_size=17),
                     A.OneOf([
                     A.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
                     A.GaussNoise(always_apply=False, p=1.0, var_limit=(375, 500.0)),
                     A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 7))
                        ],p=0.3),
                    ToTensorV2()
                    ])
    return A.Compose([
                    ToTensorV2()
                    ])


def get_valid_transform(args):
    return A.Compose([
                    ToTensorV2()
                    ])


def get_test_transform(args):
    return A.Compose([
                    ToTensorV2()
                    ])