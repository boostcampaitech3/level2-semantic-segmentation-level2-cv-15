import yaml
from tqdm import tqdm
import argparse
import numpy as np
from copy_paste import CopyPaste
from coco import CocoDetectionCP
from save_inst import save_instances
import albumentations as A

def main(config):
    img_size = config['img_size']
    transform = A.Compose([
            A.RandomScale(scale_limit=(config['limit_min'], config['limit_max']), p=config['rs_p']), #LargeScaleJitter from scale of 0.1 to 2
            A.PadIfNeeded(img_size, img_size, border_mode=0), #pads with image in the center, not the top left like the paper
            A.Resize(img_size, img_size),
            CopyPaste(blend=True, sigma=config['sigma'], pct_objects_paste=config['pct_objects_paste'], p=config['cp_p']), #pct_objects_paste is a guess
        ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
    )

    data = CocoDetectionCP(
        config['data_root'], 
        config['ann_file'], 
        transform
    )

    start_num = config['start_num']
    for file_name in tqdm(range(len(data))):
        img_data = data[file_name]
        image = img_data['image']
        masks = img_data['masks']
        bboxes = img_data['bboxes']
        if len(bboxes) <= 0:
            print("Not Exist Bbox")
            continue

        empty = np.array([])
        save_instances(image, empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=None, file_name = file_name + start_num)

        boxes = np.stack([b[:4] for b in bboxes], axis=0)
        box_classes = np.array([b[-2] for b in bboxes])
  
        mask_indices = np.array([b[-1] for b in bboxes])
        show_masks = np.stack(masks, axis=-1)[..., mask_indices]

        class_names = {k: data.coco.cats[k]['name'] for k in data.coco.cats.keys()}
        save_instances(image, boxes, show_masks, box_classes, class_names, show_bbox=True, ax=None, file_name = file_name + start_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type = str, help = 'path of run configuration yaml file')

    args = parser.parse_args()

    # load yaml
    with open(args.config) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    # running
    main(config)