import numpy as np
import cv2
from typing import Dict, List, Tuple

# ---------- LBP (P=8, R=1) + uniform mapping, grid hist ----------
def _lbp_uniform_map_table():
    # Precompute mapping for 0..255
    table = np.zeros(256, dtype=np.uint8)
    def transitions(x):
        # count bit transitions in circular 8-bit pattern
        b = [(x >> i) & 1 for i in range(8)]
        return sum(b[i] != b[(i+1) % 8] for i in range(8))
    idx = 0
    for val in range(256):
        t = transitions(val)
        if t <= 2:
            # uniform: index by number of 1 bits
            table[val] = bin(val).count("1")
        else:
            table[val] = 58  # non-uniform bucket
    return table

class LBPFaceRecognizer:
    def __init__(self):
        self._LBP_TABLE = _lbp_uniform_map_table()
        
    # ---------- Histogram distance for LBP ----------
    def chi2_distance(self, a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
        # Chi-square distance cho histogram
        a = a.astype(np.float32); b = b.astype(np.float32)
        return float(0.5 * np.sum(((a - b) ** 2) / (a + b + eps)))

    def recognize_hist(self, emb: np.ndarray, db_emb: Dict[str, List[List[float]]],
                    thresh: float = 0.60, margin: float = 0.03) -> Tuple[str, float, float]:
        """
        Nearest-neighbor trên TOÀN BỘ mẫu với khoảng cách Chi-square.
        Trả về (label, best_dist, second_best). Open-set: cần best <= thresh và (second-best - best) >= margin.
        """
        best_name = "unknown"; best = 1e9; second = 1e9; second_name = "unknown"
        for name, vecs in db_emb.items():
            for v in vecs:
                d = self.chi2_distance(emb, np.asarray(v, dtype=np.float32))
                if d < best:
                    second = best
                    second_name = best_name
                    best = d
                    best_name = name
                elif d < second:
                    second = d
                    second_name = name
        if best <= thresh and (best_name == second_name or best_name != second_name and second - best >= margin):
            return best_name, best, second
        return "unknown", best, second
    
    def lbp_u8(self, image: np.ndarray) -> np.ndarray:
        """LBP uniform P=8, R=1 for a single-channel image (uint8)."""
        # Pad 1 pixel
        I = image.astype(np.uint8)
        # neighbors
        n0 = np.roll(I, -1, axis=1)     # right
        n1 = np.roll(np.roll(I, -1, axis=0), -1, axis=1)  # right-bottom
        n2 = np.roll(I, -1, axis=0)     # bottom
        n3 = np.roll(np.roll(I, -1, axis=0), 1, axis=1)   # left-bottom
        n4 = np.roll(I, 1, axis=1)      # left
        n5 = np.roll(np.roll(I, 1, axis=0), 1, axis=1)    # left-top
        n6 = np.roll(I, 1, axis=0)      # top
        n7 = np.roll(np.roll(I, 1, axis=0), -1, axis=1)   # right-top

        c = I
        code = ((n0 >= c).astype(np.uint8) << 0) | \
            ((n1 >= c).astype(np.uint8) << 1) | \
            ((n2 >= c).astype(np.uint8) << 2) | \
            ((n3 >= c).astype(np.uint8) << 3) | \
            ((n4 >= c).astype(np.uint8) << 4) | \
            ((n5 >= c).astype(np.uint8) << 5) | \
            ((n6 >= c).astype(np.uint8) << 6) | \
            ((n7 >= c).astype(np.uint8) << 7)

        return self._LBP_TABLE[code]  # 0..59 (58 non-uniform, 0..58 valid)

    def lbp_grid_hist(self, gray_crop: np.ndarray, grid=(6,6)) -> np.ndarray:
        # Chuẩn hóa kích thước về 96x96 để ổn định
        roi = cv2.resize(gray_crop, (96, 96), interpolation=cv2.INTER_LINEAR)
        lbp = self.lbp_u8(roi)
        gh, gw = grid
        cell_h, cell_w = 96 // gh, 96 // gw
        feats = []
        for gy in range(gh):
            for gx in range(gw):
                cell = lbp[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
                hist, _ = np.histogram(cell, bins=60, range=(0,60), density=False)
                feats.append(hist)
        feat = np.concatenate(feats).astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(feat) + 1e-9
        return feat / norm