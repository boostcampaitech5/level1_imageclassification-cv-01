#run for 5 hours
#https://github.com/danielgatis/rembg
from rembg import remove
from PIL import Image
from PIL import ImageFile
import os
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_path='/opt/ml/input/data/eval/images'
output_path='/opt/ml/input/data_rmbg/eval/images'
path = Path(output_path)
#parents 상위 path 없는 경우 생성, exist_ok FileExistError 무시
path.mkdir(parents=True, exist_ok=True)

for image in os.listdir(input_path):
        if image.startswith("."):
            continue

        img_path = os.path.join(input_path, image)
        output_now_path = os.path.join(output_path, image)

        input = Image.open(img_path)
        result = remove(input)
        result = result.convert("RGB")
        result.save(output_now_path)