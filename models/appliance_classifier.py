import tensorflow as tf
import pytesseract
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델 로드
model_path = 'models/mobilenet_v2.h5'
model = load_model(model_path)

# 클래스 이름 (예시로 간단히 작성)
class_names = ['air_conditioner', 'washing_machine', 'refrigerator', 'microwave', 'oven']

def classify_appliance(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    print("Image array shape:", img_array.shape)  # 이미지 배열의 형태 출력
    predictions = model.predict(img_array)
    print("Predictions:", predictions)  # 예측 결과 출력
    
    predicted_class_index = np.argmax(predictions)
    print("Predicted class index:", predicted_class_index)  # 예측 인덱스 출력
    
    if predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
    else:
        predicted_class = "Unknown"  # 예측 인덱스가 클래스 이름 리스트의 범위를 벗어나는 경우
    
    return predicted_class

def classify_appliance(image_path):
    # 이미지 열기
    img = cv2.imread(image_path)
    
    # 텍스트 인식
    text = pytesseract.image_to_string(img, lang='kor+eng')
    print("Recognized Text:", text)  # 인식된 텍스트 출력

    # 가전제품 분류
    appliances = ['에어컨', '세탁기', '냉장고', '전자레인지', '오븐']
    for appliance in appliances:
        if appliance in text:
            return appliance

    return "Unknown"