import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(args):
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