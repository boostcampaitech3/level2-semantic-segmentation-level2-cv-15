<img width="1084" alt="image" src="https://user-images.githubusercontent.com/57162812/172509568-9e97c48a-3031-45b2-bee7-acb70f38a125.png">

## Team GAN찮아요 (CV-15) 

| 팀원 | 역할 |
|:-:|:-:|
| **김규리** | augemtation/loss/scheduler실험, smp 모델 실험, augmentation 실험|
| **박정현** | mmsegmentation 실험, ensemble |
| **석진혁** | mmsegmentation 실험, augmentation/optimizer 실험, copy&paste mmsegmentation |
| **손정균** | mmsegmentation 실험, augmentation/optimizer 실험, pseudo-labeling 적용 |
| **이현진** | pytorch baseline code 구성, copy paste augmentation, loss/optimizer/augmentation 실험 smp swin large model 구축 |

## Overview

- 현재의 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. **분리수거**는 이러한 **환경 부담을 줄일 수 있는 방법** 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, **잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각**되기 때문입니다
- 따라서 우리는 사진에서 **쓰레기를 Segmentation하는 모델**을 만들어 이러한 문제점을 해결해보고자 합니다.

## Project Scheduler

<div align="center"><img src="https://user-images.githubusercontent.com/57162812/172510441-4c78808b-3dcd-4808-bdcc-c3bd718d73f4.png" width="70%"></div>


## Dataset

- **데이터 셋 구조**
    - 11개 class
        - 쓰레기 카테고리 값에 따른 분류
        - `Background`, `General trash`, `Paper`, `Paper pack`, `Metal`, `Glass`, `Plastic`, `Styrofoam`, `Plastic bag`, `Battery`, `Clothing`
    - 데이터 셋
        - 이미지 크기 $w \times h$ : $512 \times 512$
        - 전체 데이터 개수 : 3272
        - training data : validation data = 8 : 2 + stratified 5-fold
    - Annotation file
        - Format : COCO
        - images : `id`, `height`, `width`, `filename`
        - annotations : `id`, `segmentation`, `bbox`, `area`, `category_id`, `image_id`

## 평가 방법

**MIoU**
- IoU
  <div align="center"><img src="https://user-images.githubusercontent.com/57162812/172510719-87b3ef9a-8c6b-4d5e-93cb-869a969f012f.png" width="70%"></div>

- Example of IoU
  <div align="center"><img src="https://user-images.githubusercontent.com/57162812/172510832-6d79d63b-cd8d-471c-9e08-78fb2280f112.png" width="70%"></div>

- Example of MIoU
  <div align="center"><img src="https://user-images.githubusercontent.com/57162812/172510971-580314f1-0876-41d8-852d-d453d4d5e3b0.png" width="70%"></div>
  
## Experiment

### 1. EDA

- 데이터셋 불균형
    
    <div align="center"><img src="https://user-images.githubusercontent.com/57162812/172511174-9b8310d1-c674-442a-8c6c-ca7f5aa4d86d.png" width="40%"></div>
    
- 이미지 속 object의 상대적 크기 분포

    <div align="center"><img src="https://user-images.githubusercontent.com/57162812/172511195-58dcbb97-1270-4de9-853e-9d686620a5c9.png" width="40%"></div>

→ **데이터 불균형** 및 **작은 object**를 잘 잡아내는 데에 있어서 집중할 필요가 있다는 사실 확인 가능

→ 이를 해결하기 위해 Copy-Paste등의 **Augmentation**과, Dice loss, Focal loss 등의 **손실함수**를 적용하는 등의 실험을 시도

### 2. 전처리

> **Copy-Paste Augmentation**
> 
> `copy-paste augmentation` 오픈소스코드를 활용해 부족한 class에 대해서 copy paste를 진행함으로써 데이터 증강 효과를 진행해 주었다.
> <div align="center"><img src="https://user-images.githubusercontent.com/57162812/172511440-5f626be0-7a73-4872-9385-64a9f5a66ef6.png" width="60%"></div>

> **Albumentation, Torchvision Augmentaiton**
>
>  `RandomShadow`, `RandomResizedCrop`를 통해서 작은 object 및 가려져 일부만 나온 object에 대해서 학습 가능하도록 진행해 주었다.
>  <div align="center"><img src="https://user-images.githubusercontent.com/57162812/172511602-82cbbfd5-b310-4406-8a95-9a804e437fff.png" width="60%"></div>

### 3. 모델링
#### 3-1. 모델 라이브러리

> **MMSegmentation**
> <div align="center"><img src="https://user-images.githubusercontent.com/57162812/172511762-63f5d8a4-ed7c-4390-9554-e6384cc93109.png" width="80%"></div>

> **Segmentation Models Pytorch**
> <div align="center"><img src="https://user-images.githubusercontent.com/57162812/172511839-66e689e3-1119-4146-84fe-171859848224.png" width="80%"></div>

#### 3-2. Loss
- Focal Loss, Dice Loss, Cross Entropy Loss, Lovasz Loss를 사용 하여 **데이터 불균형 문제**를 완화 시킬 수 있는 조합 및 parameter 값들을 적용한 loss 사용

### 4. 실험

#### **4-1. 모델 실험**

- `mobilenet`, `resnet50` encoder는 가벼워 40epoch 기준 2시간 가량 걸려 다양한 실험을 하기에 있어서 적합한 모델임을 확인할 수 있었다.
- `deeplabv3+`
- `ocrnet`
- `PAN` smp 모델중에서 시간대비 성능이 가장 좋게 나왔음을 확인 할 수 있었다.

#### **4-2. 하이퍼파라미터 및 augmentation 실험**

- `Loss` : Focal Loss, Dice Loss, Cross Entropy Loss, Lovasz Loss, Focal + DIce Loss, Focal + CE Loss, Encoder와 Auxiliary의 Loss 또한 다르게 적용해 보았다.
- `Optimizer` : SGD, ASGD, Adam, AdamP, AdamW, RAdam, MADGRAD
- `Augmentation`
    - Color Scale : ChannelShuffle, RandomBrightnessContrast, ToGray, HueSaturationValue
    - Weather : RadnomSnow, RandomFog, RandomShadow, RandomRain
    - Geometric : (Flip, RandomRoatet90, RandomCrop), (ShiftScaleRotate, RandomResizedCrop),
    - CoarseDropOut, Gridmask, Copy-Paste

#### **4-3. Pseudo-Label** 

- 비교적 작은 모델인 uper_resnet50 모델에 pseudo-lable을 적용 시켜 확인해본 결과 pseudo-label을 적용시킨 모델의 성능이 대폭 상승한것을 확인할 수 있었다.

#### **4-4. Ensemble**

- `Soft voting` : mIoU는 비슷하지만 클래스마다 IoU가 다른 경우가 많아서 각 모델이 잘 잡는 클래스에 가중치를 주어서 soft voting ensemble을 적용해보았다.
- `Hard voting` : 성능이 좋았던 모델들에 hard voting을 적용해보았다.

## Results

- 상위 5개의 제출 파일에 대해서 hard voting

<div align="center"><img src="https://user-images.githubusercontent.com/57162812/172512045-b786ef0e-00ae-4528-8de3-5b25fc542659.png" width="40%"></div>

- `0.8071` : 6개 모델 hard voting + 특정 클래스에 가중치
    - **mmseg :** BEit + swin large, UperNet + beit pseudo, UperNet + BEit with 512 size+pseudo, Upernet +swin large, UperNet + swin base with pseudo
    - **smp** : PANet + swin large
- `0.8053` : 7개 모델(**mmseg** **HRNet** + **OCR**, **UperNet** + **BEit**, **UperNet** + **swin base**)에 클래스 별로 가중치를 준 soft voting
- `0.7876` : **smp** **PANet** + **swin large** 1fold, 3fold, 4fold hard voting
- `0.7264` : 6개 모델(0.8071의 모델과 동일) hard voting
- `0.7255` : **mmseg** **UperNet** + **swin base** pseudo labeling

## Requirements

```jsx
pip install -r requirements.txt
```

# Train.py

```python
# MMSeg
python tools/train.py {config 파일 경로} --work-dir {work_dir 경로} --seed 21

# SMP
python3 train.py --dir {custom 내 폴더이름}
```

# Inference.py

```python
# MMSeg
python tools/inference.py {config 파일 경로} {pth 파일 경로} --file_name {저장 될 csv 파일 이름}

# SMP
python3 inference.py --dir {custom_name} --model {epoch00}
```
