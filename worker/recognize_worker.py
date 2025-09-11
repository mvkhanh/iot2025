from db.face_db import FaceDB
from model.face_detector import HaarFaceDetector
from model.face_recognizer import LBPFaceRecognizer
import threading
import queue
import numpy as np
import cv2

class RecogWorker(threading.Thread):
    """
    Single worker thread to run annotate_and_encode() asynchronously.
    Keeps only the most recent frame to avoid backlog.
    """
    def __init__(self, detector: HaarFaceDetector, recognizer: LBPFaceRecognizer, db: FaceDB, use_picam: bool, led_pins: list[int], thresh: float, margin: float, detect_every_n: int, quality: int=80):
        super().__init__(daemon=True)
        self.thresh = thresh
        self.margin = margin
        self.detect_every_n = detect_every_n
        self.use_picam = use_picam
        self.led_pins = led_pins
        self.quality = quality
        self.detector = detector
        self.recognizer = recognizer
        self.db = db
        self.q = queue.Queue(maxsize=2)
        self.last_jpg = None
        self.frame_idx = 0
        self._stop = False

    def submit(self, frame_bgr: np.ndarray):
        try:
            if self.q.full():
                # drop older frame
                self.q.get_nowait()
            self.q.put_nowait(frame_bgr)
        except Exception:
            pass

    def run(self):
        while not self._stop:
            try:
                frame = self.q.get(timeout=0.5)
            except Exception:
                continue
            jpg = self.annotate_and_encode(frame, frame_idx=self.frame_idx)
            self.frame_idx += 1
            self.last_jpg = jpg

    def stop(self):
        self._stop = True
        
    def annotate_and_encode(self, frame_bgr: np.ndarray, frame_idx=0):
        static = getattr(self.__class__.annotate_and_encode, "_static",  None)
        if static is None:
            static = {'boxes' : [], 'labels': []}
            self.__class__.annotate_and_encode._static = static

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Downscale detection and add motion gating
        need_detect = (frame_idx % self.detect_every_n == 0) or (not static["boxes"])

        if need_detect:
            # --- Downscale for faster Haar (target width ~320) ---
            target_w = 320
            if gray.shape[1] > target_w:
                scale = gray.shape[1] / float(target_w)
                small = cv2.resize(gray, (int(gray.shape[1]/scale), int(gray.shape[0]/scale)), interpolation=cv2.INTER_AREA)
            else:
                scale = 1.0
                small = gray

            # --- Motion gate: skip detect if little change ---
            # prev_small = static.get('prev_small', None)
            # if prev_small is not None:
            #     m = float(np.mean(cv2.absdiff(small, prev_small)))
            # else:
            #     m = 255.0  # force detect on first run
            # static['prev_small'] = small

            # if m < 2.0 and static["boxes"]:
            #     need_detect = False  # reuse previous boxes/labels

        if need_detect:
            faces_small = self.detector.detect(small)
            faces = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale)) for (x,y,w,h) in faces_small]
            labels = []
            for (x, y, w, h) in faces:
                # bỏ các mặt quá nhỏ để giảm nhận nhầm
                if min(w, h) < 48:
                    labels.append(("unknown", 1.0))
                    continue
                # margin một chút
                mx = int(0.10 * w); my = int(0.10 * h)
                xs = max(0, x - mx); ys = max(0, y - my)
                xe = min(frame_bgr.shape[1] - 1, x + w + mx); ye = min(frame_bgr.shape[0] - 1, y + h + my)
                # căn mắt ngang nếu có thể
                aligned = self.detector.align_face_by_eyes(gray, xs, ys, xe - xs, ye - ys)
                if aligned.size == 0:
                    labels.append(("unknown", 1.0)); continue
                proc = self.detector.preprocess_face_gray(aligned)
                emb = self.recognizer.lbp_grid_hist(proc, grid=(6, 6))
                name, dist, dist2 = self.recognizer.recognize_hist(emb, self.db.emb, thresh=float(self.thresh), margin=self.margin)
                labels.append((name, dist))
            static["boxes"] = faces
            static["labels"] = labels

        if self.use_picam:
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

        # vẽ khung + label
        for (x,y,w,h), (name, dist) in zip(static["boxes"], static["labels"]):
            color = (0, 200, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame_bgr, (x,y), (x+w, y+h), color, 2)
            txt = f"{name} ({dist:.2f})" if name != "unknown" else "unknown"
            cv2.putText(frame_bgr, txt, (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        ret, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        return jpg.tobytes() if ret else None