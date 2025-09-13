from typing import Optional
import time
import numpy as np
import cv2
from model.face_detector import HaarFaceDetector
from model.face_recognizer import LBPFaceRecognizer
from db.face_db import FaceDB

# ---------- Video source: Picamera2 (ưu tiên), fallback OpenCV ----------
class VideoSource:
    def __init__(self, width=640, height=480, fps=15, use_picam=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.picam = None
        self.use_picam = use_picam
        
        if use_picam:
            self._init_picam()
        else:
            self._init_cam()
            
    def _init_picam(self):
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            cfg = self.picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam.configure(cfg)
            self.picam.start()
        except Exception as e:
            print(f"[WARN] Picamera2 không dùng được ({e}), fallback VideoCapture(0)")
            self.use_picam = False
            self._init_cam()
    
    def _init_cam(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cam.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap = cam

    def read(self) -> Optional[np.ndarray]:
        if self.use_picam:
            # return True, cv2.cvtColor(self.picam.capture_array(), cv2.COLOR_RGB2BGR)
            return True, self.picam.capture_array()
        return self.cap.read()

    def release(self):
        if self.use_picam:
            try:
                if self.picam:
                    self.picam.stop()
            except:
                pass
        else:
            self.cap.release()
        
# ---------- CLI enroll từ camera ----------
def enroll_from_camera(name: str, num: int, cam: VideoSource, detector: HaarFaceDetector, 
                       recognizer: LBPFaceRecognizer, db: FaceDB):
    print(f"[Enroll] Thu {num} mẫu cho '{name}'. Nhấn Ctrl+C để hủy.")
    collected = 0
    time.sleep(1)
    try:
        while collected < num:
            _, frame = cam.read()
            if frame is None:
                time.sleep(0.05); continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detect(gray)
            if len(faces) == 0:
                continue
            (x,y,w,h) = max(faces, key=lambda b: b[2]*b[3])
            mx = int(0.10*w); my = int(0.10*h)
            xs = max(0, x-mx); ys = max(0, y-my)
            xe = min(frame.shape[1]-1, x+w+mx); ye = min(frame.shape[0]-1, y+h+my)

            face_crop_gray = gray[ys:ye, xs:xe]
            if face_crop_gray.size == 0:
                continue

            proc = detector.preprocess_face_gray(face_crop_gray)
            emb = recognizer.lbp_grid_hist(proc, grid=(6,6))
            db.add_embedding(name, emb, save_image=frame)
            collected += 1
            print(f"[Enroll] Collected {collected}/{num}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Enroll] Hủy bởi người dùng.")
    finally:
        cam.release()
    print(f"[Enroll] Hoàn tất. Tổng mẫu: {collected}")