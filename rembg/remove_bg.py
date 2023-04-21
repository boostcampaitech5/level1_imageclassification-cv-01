#run for 7 hours
#https://github.com/danielgatis/rembg
from rembg import remove
from PIL import Image
from PIL import ImageFile
import os
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_path='/opt/ml/input/data/train/images'
output_path='/opt/ml/input/data_edit/train/images'
path = Path(output_path)
#parents 상위 path 없는 경우 생성, exist_ok FileExistError 무시
path.mkdir(parents=True, exist_ok=True)

for profile in os.listdir(input_path):
    if profile.startswith("."): #비정상 파일 무시
        continue
    img_folder = os.path.join(input_path, profile)
        
    os.mkdir(os.path.join(output_path, profile)) #새로 이미지가 들어갈 파일

    for image in os.listdir(img_folder):
        if image.startswith("."):
            continue

        img_path = os.path.join(input_path, profile, image)
        output_now_path = os.path.join(output_path, profile, image)

        input = Image.open(img_path)
        result = remove(input)
        result = result.convert("RGB")
        result.save(output_now_path)