import os
import os.path as osp
import time
import math
import random
from datetime import timedelta
from argparse import ArgumentParser

from model import load_model
from utils import collate_fn, save_model, label_accuracy_score, add_hist, set_seed, sort_class
from augmentation import get_test_transform, get_train_transform, get_valid_transform
from dataset import CustomDataLoader
from validation import validation
from test import test

from utils import label_accuracy_score, add_hist
import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from importlib import import_module
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb

from loss import create_criterion
from optimizer import create_optimizer
from scheduler import create_scheduler

def parse_args():
    parser = ArgumentParser()

    # Conventional args

    parser.add_argument('--inference', type=bool, default=False)
    parser.add_argument('--model_path', type=str)

    parser.add_argument('--dataset_path', type=str, default="/opt/ml/input/data")
    parser.add_argument('--saved_dir', type=str, default = "/opt/ml/input/code/saved")
    parser.add_argument('--train_name', type=str,default="splited/train_0.json") 
    parser.add_argument('--valid_name', type=str, default="splited/valid_0.json")
    parser.add_argument('--test_name', type=str, default="test.json")
 
    parser.add_argument('--project', type=str, default = "segmentation")
    parser.add_argument('--entity', type=str, default ="boostcampaitech3")
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--print_step', type=int, default = 25)
    parser.add_argument('--seed', type=int, default=21)

    parser.add_argument('--model', type=str, default="fcn_resnet50")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type')

    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer type (default: adam)")
    parser.add_argument('--learning_rate', type=int, default=0.0001)
    parser.add_argument('--weight_decay', type=int, default = 1e-6)
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum (default: 0.9)" )

    parser.add_argument("--scheduler", type=str, default="lambda", help="scheduler type (default: lambda)")
    parser.add_argument("--poly_exp", type=float, default=1.0, help="polynomial LR exponent (default: 1.0)",)
    parser.add_argument("--T_max", type=int, default=10, help="cosineannealing T_max (default: 10)")
    parser.add_argument("--eta_min", type=int, default=0, help="cosineannealing eta_min (default: 0)")
    parser.add_argument("--step_size", type=int, default=10, help="stepLR step_size (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.1, help="stepLR gamma (default: 0.1)")

    parser.add_argument('--exp_name', type=str)
    parser.add_argument("--vis_every", type=int, default=10, help="image logging interval")
    
    args = parser.parse_args()

    return args

def do_training(args):
    
    exp_name =args.exp_name
    
    train_path = args.dataset_path + '/' + args.train_name
    val_path = args.dataset_path + '/' + args.valid_name
    test_path = args.dataset_path + '/' + args.test_name
    device = args.device
    sorted_df = sort_class()

    if not args.inference:
        wandb.init(project=args.project, entity=args.entity, name = exp_name)

        train_dataset = CustomDataLoader(dataset_path = args.dataset_path, data_dir=train_path, mode='train', transform=get_train_transform(args), sorted_df = sorted_df)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

        val_dataset = CustomDataLoader(dataset_path = args.dataset_path, data_dir=val_path, mode='val', transform=get_valid_transform(args), sorted_df = sorted_df)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
        
        model = load_model(args.model, args.num_classes)
        wandb.watch(model)
        
        best_mIoU = 0

        criterion = create_criterion(args.criterion)
        print(criterion)
        optimizer = create_optimizer(args, model)
        scheduler = create_scheduler(args, optimizer)
        
        for epoch in range(args.num_epoch):
            model.train()

            hist = np.zeros((args.num_classes, args.num_classes))
            for step, (images, masks, _) in enumerate(tqdm(train_loader, total = len(train_loader))):
                images = torch.stack(images)
                masks = torch.stack(masks).long()

                # gpu 연산을 위해 device 할당
                images, masks = images.to(device), masks.to(device)
                
                # device 할당
                model = model.to(device)
                
                # inference
                outputs = model(images)['out']

                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=args.num_classes)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

                # step 주기에 따른 loss 출력
                if (step + 1) % args.print_step == 0:
                    print(f'Epoch [{epoch+1}/{args.num_epoch}], Step [{step+1}/{len(train_loader)}], \
                            Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                    wandb.log({"train/loss": round(loss.item(),4)})
            
            pre_mIoU = validation(epoch + 1, model, val_loader, criterion, device, train_path, sorted_df, args.vis_every)
            # avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            
            if pre_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {args.saved_dir}")
                best_mIoU = pre_mIoU
                save_model(model, args.saved_dir, exp_name)
        # scheduler.step()
        return

    #### INFERENCE ####

    # test dataset loading
    test_dataset = CustomDataLoader(dataset_path = args.dataset_path, data_dir=test_path, mode='test', transform=get_test_transform(args), sorted_df = sorted_df)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn)

    # best model 저장 경러
    model_path = args.model_path

    # best model 불러오기
    model = load_model(args.model, args.num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.classifier[-1].out_channels = args.num_classes

    # sample_submisson.csv 열기
    submission = pd.read_csv(f'/opt/ml/input/code/submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(f"/opt/ml/input/code/submission/{exp_name}.csv", index=False)


def main(args):
    set_seed(args)
    do_training(args)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
