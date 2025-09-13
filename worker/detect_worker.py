from db.face_db import FaceDB
from model.face_detector import HaarFaceDetector
from model.face_recognizer import LBPFaceRecognizer
import threading
import queue
import numpy as np
import cv2

class DetectWorker(threading.Thread):
    """
    Single worker thread to run annotate_and_encode() asynchronously.
    Keeps only the most recent frame to avoid backlog.
    """
    def __init__(self, detector: HaarFaceDetector, use_picam: bool, led_pins: list[int], thresh: float, margin: float, detect_every_n: int, quality: int=80):
        super().__init__(daemon=True)
        self.thresh = thresh
        self.margin = margin
        self.detect_every_n = detect_every_n
        self.use_picam = use_picam
        self.led_pins = led_pins
        self.quality = quality
        self.detector = detector
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
            static = {'boxes' : []}
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
            faces = [(x, y, w, h) for (x, y, w, h) in faces if min(w, h) >= 48]
            static["boxes"] = faces

        if self.use_picam and len(self.led_pins) > 0:
            import RPi.GPIO as GPIO
            # Sáng đèn nào đó nếu số người vượt mức
            if len(static['boxes']):
                GPIO.output(self.led_pins[0], GPIO.HIGH)
            else:
                GPIO.output(self.led_pins[0], GPIO.LOW)

        # vẽ khung + label
        for (x,y,w,h) in zip(static["boxes"]):
            color = (0, 255, 0)
            cv2.rectangle(frame_bgr, (x,y), (x+w, y+h), color, 2)

        ret, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        return jpg if ret else None