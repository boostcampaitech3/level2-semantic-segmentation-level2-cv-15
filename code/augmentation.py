import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from copy_paste import CopyPaste

def get_train_transform(args):
    if not args.copy_paste:
        return A.Compose([
                        ToTensorV2()
                        ])
    else:
        return A.Compose([
                    CopyPaste(args, blend=False, sigma=1, pct_objects_paste=args.copy_pct, p=args.copy_p), #pct_objects_paste is a guess
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
            )


def get_valid_transform(args):
    return A.Compose([
                    ToTensorV2()
                    ])


def get_test_transform(args):
    return A.Compose([
                    ToTensorV2()
                    ])