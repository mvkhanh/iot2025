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
    
    def detect(self, gray: np.ndarray) -> List[Tuple[int,int,int,int]]:
        faces = self.det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]