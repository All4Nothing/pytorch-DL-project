import io
import json
# request 라이브러리로 flask 서버에 POST 요청을 생성할 수 있다.
import requests
from PIL import Image

from torchvision import transforms


image = Image.open("./digit_image.jpg")

def image_to_tensor(image):
    gray_image = transforms.functional.to_grayscale(image)
    resized_image = transforms.functional.resize(gray_image, (28, 28))
    input_image_tensor = transforms.functional.to_tensor(resized_image)
    input_image_tensor_norm = transforms.functional.normalize(input_image_tensor, (0.1302,), (0.3069,))
    return input_image_tensor_norm

image_tensor = image_to_tensor(image)

dimensions = io.StringIO(json.dumps({'dims': list(image_tensor.shape)}))
data = io.BytesIO(bytearray(image_tensor.numpy()))

# `r`은 flask 서버에 보내는 request에 대한 response(모델 예측)를 수신한다. 
r = requests.post('http://localhost:8890/test',
                  files={'metadata': dimensions, 'data' : data})

response = json.loads(r.content)

print("Predicted digit :", response)