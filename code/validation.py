import torch
import numpy as np
from utils import label_accuracy_score, add_hist, sort_class
import wandb
from tqdm import tqdm

from mlflow import log_metric, log_param, log_artifacts, log_params, log_metrics

class_labels = {
    0: "Backgroud",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}
def validation(epoch, model, data_loader, criterion, device, train_path, sorted_df, interval):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(tqdm(data_loader, total=len(data_loader))):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            if step % interval == 0:
                    wandb.log(
                        {
                            "visualize": wandb.Image(
                                images[0, :, :, :],
                                masks={
                                    "predictions": {
                                        "mask_data": outputs[0, :, :],
                                        "class_labels": class_labels,
                                    },
                                    "ground_truth": {
                                        "mask_data": masks[0, :, :],
                                        "class_labels": class_labels,
                                    },
                                },
                            )
                        }
                    )
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , sorted_df['Categories'])]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        log = {
            "val/mIoU": round(mIoU, 4),
            "val/loss": round(avrg_loss.item(), 4),
            "val/accuracy": round(acc, 4),
        }
        for d in IoU_by_class:
            for cls in d:
                log[f"val/{cls}_IoU"] = d[cls]
        wandb.log(log)
        log_metrics(log)
        
    # return avrg_loss
    return round(mIoU, 4)
