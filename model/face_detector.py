import os
from typing import List, Tuple
import numpy as np
import cv2

class HaarFaceDetector:
    def __init__(self, use_haar_eye=True):
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(haar_path):
            raise RuntimeError("Không tìm thấy haarcascade_frontalface_default.xml")
        self.det = cv2.CascadeClassifier(haar_path)
        
        self.eye_det = None
        if use_haar_eye:
            haar_eye_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
            if not os.path.exists(haar_eye_path):
                raise RuntimeError("Không tìm thấy haarcascade_eye_tree_eyeglasses.xml")
            self.eye_det = cv2.CascadeClassifier(haar_eye_path)
            
        # ---------- Illumination normalization (CLAHE + gamma) ----------
        self.clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.gamma_lut = (np.clip(((np.arange(256) / 255.0) ** 0.8) * 255.0, 0, 255)).astype(np.uint8)

    def preprocess_face_gray(self, gray_crop: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa ánh sáng để LBP ổn định hơn: CLAHE + gamma 0.8 + blur nhẹ.
        Đầu vào/ra: ảnh GRAY.
        """
        if gray_crop.size == 0:
            return gray_crop
        # resize ổn định để CLAHE ổn định (dù lbp_grid_hist cũng sẽ resize)
        g = cv2.resize(gray_crop, (96, 96), interpolation=cv2.INTER_LINEAR)
        g = self.clache.apply(g)
        g = cv2.LUT(g, self.gamma_lut)
        # blur nhẹ để giảm nhiễu muối tiêu
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return g
    
    def align_face_by_eyes(self, gray_full: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Căn mắt ngang đơn giản bằng Haar eye trong ROI. Nếu không phát hiện được mắt, trả về ROI như cũ.
        gray_full: ảnh xám toàn khung
        """
        roi = gray_full[y:y+h, x:x+w]
        if self.eye_det is None or roi.size == 0:
            return roi
        eyes = self.eye_det.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=3, minSize=(12, 12))
        if len(eyes) < 2:
            return roi
        # lấy 2 mắt theo x nhỏ -> lớn
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes[0], eyes[1]
        p1 = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
        p2 = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        M = cv2.getRotationMatrix2D((x + w // 2, y + h // 2), angle, 1.0)
        aligned = cv2.warpAffine(gray_full, M, (gray_full.shape[1], gray_full.shape[0]), flags=cv2.INTER_LINEAR)
        return aligned[y:y+h, x:x+w]
    
    def detect(self, gray: np.ndarray) -> List[Tuple[int,int,int,int]]:
        faces = self.det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]