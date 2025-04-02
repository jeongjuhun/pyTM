import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps, ImageFont, ImageDraw

np.set_printoptions(suppress=True)

# 모델 및 라벨 로드
model = load_model(r"C:\2025_AI\pjt005\model\keras_model.h5", compile=False)
class_names = open(r"C:\2025_AI\pjt005\model\labels.txt", "r", encoding="utf-8").readlines()

# 한글 폰트 설정 (윈도우 기본 폰트 사용)
fontpath = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(fontpath, 30)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # 모델 입력 크기에 맞게 리사이즈
    size = (224, 224)
    resized_image = ImageOps.fit(pil_image, size, Image.Resampling.LANCZOS)

    # 이미지 배열 변환 및 정규화
    image_array = np.asarray(resized_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    # 예측
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    result_text = f'클래스: {class_name[2:]} | 신뢰도: {confidence_score:.2f}'

    # PIL로 한글 텍스트 추가
    draw = ImageDraw.Draw(pil_image)
    draw.text((10, 10), result_text, font=font, fill=(0, 255, 0))

    # 다시 OpenCV 형식으로 변환하여 출력
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.imshow("실시간 영상 분류", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
