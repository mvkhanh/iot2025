import os, json
import cv2
import numpy as np
from typing import Tuple, Dict

class LBPHRecognizer:
    """
    Wrapper for OpenCV cv2.face.LBPHFaceRecognizer.
    - radius=1, neighbors=8, grid=(6,6)
    - train/update từ FaceDB (ảnh xám đã crop)
    - predict trả về (name | 'unknown', distance). distance càng nhỏ càng tốt.
    """
    def __init__(self, radius: int = 1, neighbors: int = 8,
                 grid_x: int = 6, grid_y: int = 6, 
                 model_dir: str = "faces_db/model"):
        if not hasattr(cv2, "face"):
            raise RuntimeError("Cần cài opencv-contrib-python để dùng cv2.face")

        self.model = cv2.face.LBPHFaceRecognizer_create(
            radius=radius, neighbors=neighbors, grid_x=grid_x, grid_y=grid_y
        )
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "lbph.yml")
        self.labels_path = os.path.join(self.model_dir, "labels.json")

        self.id2name: Dict[int, str] = {}
        self.name2id: Dict[str, int] = {}
        self._trained = False
        self._try_load()

    # ---------- IO ----------
    def _try_load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
            self.model.read(self.model_path)
            with open(self.labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.id2name = {int(k): v for k, v in data.items()}
            self.name2id = {v: k for k, v in self.id2name.items()}
            self._trained = True

    def _save(self):
        self.model.write(self.model_path)
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump(self.id2name, f, ensure_ascii=False)

    # ---------- API ----------
    def train_from_facedb(self, facedb) -> None:
        imgs, labels = facedb.get_training_arrays()
        if not imgs:
            raise ValueError("FaceDB trống: chưa có ảnh để train.")
        self.model.train(imgs, np.asarray(labels, dtype=np.int32))
        self.id2name = facedb.id2name_copy()
        self.name2id = {v: k for k, v in self.id2name.items()}
        self._trained = True
        self._save()

    def update_from_facedb(self, facedb) -> None:
        imgs, labels = facedb.get_training_arrays()
        if not imgs:
            return
        if self._trained:
            self.model.update(imgs, np.asarray(labels, dtype=np.int32))
        else:
            self.model.train(imgs, np.asarray(labels, dtype=np.int32))
            self._trained = True
        self.id2name = facedb.id2name_copy()
        self.name2id = {v: k for k, v in self.id2name.items()}
        self._save()

    def predict(self, gray_crop, thresh: float = 60.0) -> Tuple[str, float]:
        """
        LBPH trả về (label_id, confidence). confidence là khoảng cách (càng nhỏ càng tốt).
        Open-set đơn giản: nếu conf <= thresh -> trả về name, ngược lại 'unknown'.
        """
        if not self._trained:
            return "unknown", float("inf")
        lab_id, conf = self.model.predict(gray_crop)
        name = self.id2name.get(lab_id, "unknown")
        if conf <= thresh and name != "unknown":
            return name, float(conf)
        return "unknown", float(conf)