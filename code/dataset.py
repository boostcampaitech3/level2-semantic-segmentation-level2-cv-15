from torch.utils.data import Dataset
from pycocotools.coco import COCO
from coco import CocoDetectionCP
from copy_paste import copy_paste_class
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, dataset_path, data_dir, sorted_df, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.dataset_path = dataset_path
        self.sorted_df = sorted_df
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                category_names = list(self.sorted_df.Categories)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


## COPY_PASTE
class CustomCPDataLoader(Dataset):
    """COCO format"""
    def __init__(self, dataset_path, data_dir, sorted_df, mode = 'train', transforms = None):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.coco = COCO(data_dir)
        self.dataset_path = dataset_path
        self.data_dir = data_dir
        self.sorted_df = sorted_df
        self.data = CocoDetectionCP(
                            self.dataset_path, 
                            self.data_dir, 
                            self.transforms
                        )
        print("dataset init")
        
    def __getitem__(self, index: int):
        tmp_data = self.data[index]

        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = tmp_data["image"].astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            bboxes = tmp_data['bboxes']
            box_classes = np.array([b[-2] for b in bboxes])
            mask_indices = np.array([b[-1] for b in bboxes])
            mask = np.array(tmp_data["masks"])[mask_indices]
            area = [np.sum(m) for m in mask]
            big_idx = np.argsort(area)[::-1]
            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            for i in big_idx:
                masks[mask[i] == 1] = box_classes[i]
                # print(len(mask), len(box_classes))
            masks = masks.astype(np.int8)

            bboxes = []
            for ix, obj in enumerate(anns):
                bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])
            
            # transform -> albumentations 라이브러리 활용
            transform = A.Compose([
                        ToTensorV2()
                        ])
            transformed = transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())