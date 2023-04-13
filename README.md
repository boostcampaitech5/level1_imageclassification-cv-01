# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.7.1
- torchvision==0.8.2                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`

### Commit Type
feat : 새로운 기능 추가, 기존의 기능을 요구 사항에 맞추어 수정<br>
fix : 기능에 대한 버그 수정<br>
build : 빌드 관련 수정<br>
chore : 패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore<br>
ci : CI 관련 설정 수정<br>
docs : 문서(주석) 수정<br>
style : 코드 스타일, 포맷팅에 대한 수정<br>
refactor : 기능의 변화가 아닌 코드 리팩터링 ex) 변수 이름 변경<br>
test : 테스트 코드 추가/수정<br>
release : 버전 릴리즈
