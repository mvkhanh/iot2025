import os
from typing import List, Tuple
import numpy as np
import cv2

class HaarFaceDetector:
    def __init__(self):
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(haar_path):
            raise RuntimeError("Không tìm thấy haarcascade_frontalface_default.xml")
        self.det = cv2.CascadeClassifier(haar_path)

    def preprocess_face_gray(self, gray_crop: np.ndarray) -> np.ndarray:
        """
        Đầu vào/ra: ảnh GRAY.
        """
        if gray_crop.size == 0:
            return gray_crop
        g = cv2.resize(gray_crop, (96, 96), interpolation=cv2.INTER_LINEAR)
        return g
    
    def detect(self, gray: np.ndarray) -> List[Tuple[int,int,int,int]]:
        faces = self.det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]