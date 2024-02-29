import onnxruntime
import numpy as np
import cv2

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50,thickness=(2,2,2)):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),thickness[0])
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),thickness[1])
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),thickness[2])

    return img

class FSANet:
    def __init__(self, onnx_path):
        self.ort_session = onnxruntime.InferenceSession(onnx_path)
        self.input_width = 64  # Замените на фактический размер входного изображения
        self.input_height = 64  # Замените на фактический размер входного изображения
        self.pad = 0.1  # Замените на фактическое значение
        self.input_node_dims = [1, 3, self.input_height, self.input_width]  # Порядок может быть изменен в зависимости от вашей модели
        self.input_node_names = ['input']  # Замените на фактическое имя входного узла
        self.output_node_names = ['output']  # Замените на фактическое имя выходного узла

    def transform(self, mat):
        h, w, _ = mat.shape
        nh = int((1 + self.pad) * h)
        nw = int((1 + self.pad) * w)

        nx1 = max(0, int((nw - w) / 2))
        ny1 = max(0, int((nh - h) / 2))

        canvas = np.zeros((nh, nw, 3), dtype=np.uint8)
        canvas[ny1:ny1 + h, nx1:nx1 + w, :] = mat

        canvas = cv2.resize(canvas, (self.input_width, self.input_height))
        canvas = (canvas - 127.5) / 127.5  # Нормализация

        # Создайте тензор
        input_tensor = np.expand_dims(canvas.transpose((2, 0, 1)), axis=0).astype(np.float32)

        return input_tensor

    def detect(self, mat):
        input_tensor = self.transform(mat)

        # Выполните предсказание
        output_tensors = self.ort_session.run(self.output_node_names, {'input': input_tensor})

        angles_ptr = output_tensors[0]

        yaw, pitch, roll = np.split(angles_ptr.flatten(), 3)
        
        euler_angles = {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'flag': True
        }

        return euler_angles

