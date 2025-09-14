import os, json, glob, threading
from typing import Dict, List, Optional
from datetime import datetime
import cv2

class FaceDB:
    """
    Kho lưu mẫu khuôn mặt dạng ảnh xám:
      - Ảnh mẫu: faces_db/samples/<name>/*.png
      - Metadata map name<->id: faces_db/meta.json
    Cung cấp API để thêm mẫu, liệt kê danh tính, và xuất (imgs, labels) cho LBPH.
    """
    def __init__(self, db_dir: str = "faces_db"):
        self.db_dir = db_dir
        self.samples_dir = os.path.join(db_dir, "samples")
        self.meta_path = os.path.join(db_dir, "meta.json")
        os.makedirs(self.samples_dir, exist_ok=True)

        self.lock = threading.Lock()
        self.name2id: Dict[str, int] = {}
        self.id2name: Dict[int, str] = {}
        self._load_meta()

    # ---------- meta ----------
    def _load_meta(self) -> None:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # Chuẩn hoá kiểu dữ liệu
            self.name2id = {str(k): int(v) for k, v in obj.get("name2id", {}).items()}
            self.id2name = {int(k): str(v) for k, v in obj.get("id2name", {}).items()}
        else:
            self.name2id, self.id2name = {}, {}

    def _save_meta(self) -> None:
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"name2id": self.name2id, "id2name": self.id2name}, f, ensure_ascii=False)
        os.replace(tmp, self.meta_path)

    def ensure_label(self, name: str) -> int:
        """
        Đảm bảo có id cho 'name', nếu chưa có thì cấp mới (1..N) và lưu meta.
        """
        with self.lock:
            if name not in self.name2id:
                new_id = len(self.name2id) + 1
                self.name2id[name] = new_id
                self.id2name[new_id] = name
                self._save_meta()
            return self.name2id[name]

    def list_identities(self) -> List[str]:
        return sorted(self.name2id.keys())

    # ---------- dữ liệu ảnh ----------
    def add_sample(self, name: str, gray_crop, save_full_bgr: Optional[cv2.Mat] = None) -> str:
        """
        Thêm 1 ảnh xám (crop khuôn mặt) vào kho. Trả về đường dẫn ảnh đã lưu.
        Tuỳ chọn: save_full_bgr để lưu cả khung hình gốc phục vụ kiểm chứng.
        """
        _ = self.ensure_label(name)
        user_dir = os.path.join(self.samples_dir, name)
        os.makedirs(user_dir, exist_ok=True)
        idx = len(glob.glob(os.path.join(user_dir, "*.png")))
        path = os.path.join(user_dir, f"{idx:04d}.png")
        cv2.imwrite(path, gray_crop)

        if save_full_bgr is not None:
            shots_dir = os.path.join(self.db_dir, "shots", name)
            os.makedirs(shots_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(shots_dir, f"{ts}.jpg"), save_full_bgr)
        return path

    def get_training_arrays(self):
        """
        Trả về (imgs, labels) để train LBPH:
          - imgs: list[np.ndarray(HxW, uint8)]
          - labels: list[int]
        """
        imgs, labels = [], []
        for name, lid in self.name2id.items():
            user_dir = os.path.join(self.samples_dir, name)
            for p in sorted(glob.glob(os.path.join(user_dir, "*.png"))):
                g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                imgs.append(g)
                labels.append(lid)
        return imgs, labels

    def id2name_copy(self) -> Dict[int, str]:
        return dict(self.id2name)
