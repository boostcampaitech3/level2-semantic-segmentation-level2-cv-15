# Install

1. 가상환경 구축하기

```
conda create -n mmseg python=3.7 -y
conda activate mmseg
```


2. PyTorch, Torchvision, mmcv 설치

```
conda install pytorch=1.11 torchvision cudatoolkit=10.2 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11/index.html
```

3. MMSegmentation 설치

```
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .
```

4. JupyterLab에 가상환경 설치
```
pip install ipykernel
python -m ipykernel install --user --name mmseg
```



---

# 데이터 형식 변경

1. 다음과 같이 폴더를 만들고 make_json_image.py, make_json_mask.ipynb, split_txt.ipynb를 넣어주세요.
```
├──input
|   ├──data─┬──make_json_image.py
|   |       ├──make_json_mask.ipynb
|   |       └──copy_paste
|   |            ├──__init__.py
|   |            ├──coco.py
|   |            ├──copy_paste_main.py
|   |            ├──copy_paste.py
|   |            ├──save_inst.py
|   |            └──cp_config.yaml
|   └──mmseg               
|       ├──images
|       ├──annotations
|       ├──test 
|       └──split_txt.ipynb
└──mmsegmentation
```

2. input/data/make_json_image.py의 9번 라인에 전체 데이터 json 파일 경로를 수정한 후 실행

3. input/data/make_json_mask.ipynb의 2번째 셀에 전체 데이터 json 파일 경로를 수정한 후 실행

4. input/mmseg/split_txt.ipynb에서 splited 데이터 json 파일 경로를 수정한 후 실행

---

# Train 

```
python tools/train.py {config 파일 경로} --work-dir {work_dir 경로} --seed 21
```

# Inference
1. mmsegmentation/output 폴더에 code/submission/sample_submission.csv파일을 복사

2. inference.py 파일을 tools 폴더에 넣은 후 아래와 같이 실행

```
python tools/inference.py {config 파일 경로} {pth 파일 경로} --file_name {저장 될 csv 파일 이름}
```

# Copy & Paste

1. copy_paste 폴더로 이동

2. cp_config.yaml의 복사하고 싶은 class를 cls_ids에 list형식으로 추가

3. copy_paste_main.py를 아래와 같이 실행

```
python copy_paste_main.py --config {config 파일 경로}
```

4. input/mmseg/images와 input/mmseg/annotations에 다음과 같이 저장  
ex) {추가한class}_{파일번호}.jpg,  {추가한class}_{파일번호}.png

5. 복사하고 싶은 json파일은 cp_config.yaml에서 추가할 수 있고, input/mmseg에 json과 같은 파일명이 txt로 저장


# Soft voting ensemble
#### 1. 코드 수정
mmseg/models/segmentors/encoder_decoder.py의 코드를 다음과 같이 수정
- TTA가 있을 경우 : aug_test 함수 내부 (284번 line)
- TTA가 없을 경우 : simple_test 함수 내부 (261번 line)

```
seg_pred = seg_logit.argmax(dim=1)   # 원래 코드
seg_pred = seg_logit                 # 수정 코드
```

#### 2. Inference
```
python tools/inference_soft.py {config 파일 경로} {pth 파일 경로} --file_name {저장 될 bin 파일 이름}
```
