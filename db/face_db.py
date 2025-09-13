import os, json
from datetime import datetime
import threading
from typing import Dict, List, Optional
import numpy as np
import cv2

# ---------- FaceDB: lưu embeddings (LBP grid hist) ----------
class FaceDB:
    def __init__(self, db_dir="faces_db"):
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.emb: Dict[str, List[List[float]]] = {}  # name -> list of embeddings
        self._load()

    @property
    def json_path(self):
        return os.path.join(self.db_dir, "embeddings.json")

    def _load(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                self.emb = json.load(f)
        else:
            self.emb = {}

    def _save(self):
        tmp = self.json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.emb, f)
        os.replace(tmp, self.json_path)

    def list_identities(self):
        return sorted(list(self.emb.keys()))

    def add_embedding(self, name: str, vec: np.ndarray, save_image: Optional[np.ndarray] = None):
        with self.lock:
            if name not in self.emb:
                self.emb[name] = []
            self.emb[name].append(vec.tolist())
            self._save()
        # lưu ảnh gốc (nếu có)
        if save_image is not None:
            person_dir = os.path.join(self.db_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(person_dir, f"{ts}.jpg"), save_image)
