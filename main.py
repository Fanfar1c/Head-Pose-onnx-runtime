import numpy as np
import cv2
from fsanet_util import FSANet, draw_axis

var_onnx_path = "fsanet-var.onnx"  # Замените на фактический путь к вашей ONNX-модели
conv_onnx_path = "fsanet-1x1.onnx"  # Замените на фактический путь к вашей ONNX-модели
var_fsanet = FSANet(var_onnx_path)
conv_fsanet = FSANet(conv_onnx_path)

cap = cv2.VideoCapture(0)

while True:
    # Получение кадра с веб-камеры
    ret, frame = cap.read()
    
    if not ret:
        print("Не удалось получить кадр с веб-камеры.")
        continue
    
    image = frame

# Выполните детекцию
    var_euler_angles = var_fsanet.detect(image)
    conv_euler_angles = conv_fsanet.detect(image)

    euler_angles = {
        'yaw': (var_euler_angles['yaw'] + conv_euler_angles['yaw']) / 2.0,
        'pitch': (var_euler_angles['pitch'] + conv_euler_angles['pitch']) / 2.0,
        'roll': (var_euler_angles['roll'] + conv_euler_angles['roll']) / 2.0,
        'flag': var_euler_angles['flag'] and conv_euler_angles['flag']
    }

    imagedata = draw_axis(image,euler_angles['yaw'],euler_angles['pitch'],euler_angles['roll'])
    
    image = np.array(imagedata, dtype=np.uint8)
    

    
    cv2.imshow('Web Camera Face Detection', image)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
