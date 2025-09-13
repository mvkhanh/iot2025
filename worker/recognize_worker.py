import numpy as np
import cv2

from db.face_db import FaceDB
from worker.detect_worker import DetectWorker

class RecogWorker(DetectWorker):
    """
    Multiprocessing worker for detection + recognition.
    Heavy objects (detector, recognizer, DB) are constructed inside the child process.
    """
    def __init__(self, detector_cls, recognizer_cls, use_picam: bool, led_pins: list[int],
                 thresh: float, margin: float, detect_every_n: int, quality: int = 80,
                 in_q=None, out_q=None):
        super().__init__(detector_cls=detector_cls, use_picam=use_picam, led_pins=led_pins,
                         detect_every_n=detect_every_n, quality=quality, in_q=in_q, out_q=out_q)
        self.recognizer_cls = recognizer_cls
        self.thresh = thresh
        self.margin = margin

    def _init_in_child(self):
        # Build detector, recognizer, and DB in child process
        self.detector = self.detector_cls()
        self.recognizer = self.recognizer_cls()
        self.db = FaceDB()

    def annotate_and_encode(self, frame_bgr: np.ndarray, frame_idx = 0, static=None):
        if static is None:
            static = {"boxes": [], "labels": []}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        need_detect = (frame_idx % self.detect_every_n == 0) or (not static["boxes"])

        if need_detect:
            scale = 1.0
            small = gray

            faces_small = self.detector.detect(small)
            faces = [(int(x * scale), int(y * scale), int(w * scale), int(h * scale))
                     for (x, y, w, h) in faces_small]
            labels = []
            for (x, y, w, h) in faces:
                if min(w, h) < 48:
                    labels.append(("unknown", 1.0))
                    continue
                mx = int(0.10 * w); my = int(0.10 * h)
                xs = max(0, x - mx); ys = max(0, y - my)
                xe = min(frame_bgr.shape[1] - 1, x + w + mx); ye = min(frame_bgr.shape[0] - 1, y + h + my)
                aligned = self.detector.align_face_by_eyes(gray, xs, ys, xe - xs, ye - ys)
                if aligned.size == 0:
                    labels.append(("unknown", 1.0)); continue
                proc = self.detector.preprocess_face_gray(aligned)
                emb = self.recognizer.lbp_grid_hist(proc, grid=(6, 6))
                name, dist, dist2 = self.recognizer.recognize_hist(
                    emb, self.db.emb, thresh=float(self.thresh), margin=self.margin
                )
                labels.append((name, dist))
            static["boxes"] = faces
            static["labels"] = labels

        # LED per-identity feedback on Pi (best-effort)
        if self.use_picam:
            try:
                import RPi.GPIO as GPIO
                identities = self.db.list_identities()
                active = set()
                for (lb, dist) in static['labels']:
                    if lb != "unknown" and lb in identities:
                        idx = identities.index(lb)
                        if idx < len(self.led_pins):
                            GPIO.output(self.led_pins[idx], GPIO.HIGH)
                            active.add(idx)
                for i in range(min(len(self.led_pins), len(identities))):
                    if i not in active:
                        GPIO.output(self.led_pins[i], GPIO.LOW)
            except Exception:
                pass

        for (x, y, w, h), (name, dist) in zip(static["boxes"], static["labels"]):
            color = (0, 200, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
            txt = f"{name} ({dist:.2f})" if name != "unknown" else "unknown"
            cv2.putText(frame_bgr, txt, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return frame_bgr