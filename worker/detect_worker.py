from multiprocessing import Process, Queue, Event
import queue as pyqueue
import numpy as np
import cv2
from typing import Optional

# NOTE: detector must be constructed inside the child process to avoid pickling issues.
class DetectWorker(Process):
    """
    Multiprocessing worker: receive frames via in_q, run detection, send annotated frames via out_q.
    Keeps only the newest frame in both queues to avoid backlog.
    """
    def __init__(self, detector_cls, use_picam: bool, led_pins: list[int],
                 detect_every_n: int, quality: int = 80,
                 in_q: Optional[Queue] = None, out_q: Optional[Queue] = None):
        super().__init__(daemon=True)
        self.detector_cls = detector_cls
        self.use_picam = use_picam
        self.led_pins = led_pins
        self.quality = quality
        self.detect_every_n = detect_every_n
        self.in_q = in_q or Queue(maxsize=2)
        self.out_q = out_q or Queue(maxsize=1)
        self._stop = Event()
        self.frame_idx = 0

    # ----- API for the main process -----
    def submit(self, frame_bgr: np.ndarray):
        try:
            if self.in_q.full():
                _ = self.in_q.get_nowait()  # drop oldest
            self.in_q.put_nowait(frame_bgr)
        except Exception:
            pass

    def poll(self):
        """Drain out_q and return the latest annotated frame (or None)."""
        last = None
        try:
            while True:
                last = self.out_q.get_nowait()
        except pyqueue.Empty:
            return last

    def stop(self):
        self._stop.set()
        try:
            self.in_q.put_nowait(None)
        except Exception:
            pass

    # ----- Child process lifecycle -----
    def _init_in_child(self):
        # Construct heavy objects in child process
        self.detector = self.detector_cls()

    def run(self):
        self._init_in_child()
        static = {"boxes": []}
        while not self._stop.is_set():
            try:
                frame = self.in_q.get(timeout=0.5)
            except pyqueue.Empty:
                continue
            if frame is None:
                continue

            out = self.annotate_and_encode(frame, frame_idx=self.frame_idx, static=static)
            self.frame_idx += 1

            if out is not None:
                # keep only the newest output
                try:
                    while True:
                        self.out_q.get_nowait()
                except pyqueue.Empty:
                    pass
                try:
                    self.out_q.put_nowait(out)
                except Exception:
                    pass

    # ----- Detection & drawing -----
    def annotate_and_encode(self, frame_bgr: np.ndarray, frame_idx = 0, static=None):
        if static is None:
            static = {"boxes": []}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        need_detect = (frame_idx % self.detect_every_n == 0) or (not static["boxes"])

        if need_detect:
            scale = 1.0
            small = gray

            faces_small = self.detector.detect(small)
            faces = [(int(x * scale), int(y * scale), int(w * scale), int(h * scale))
                     for (x, y, w, h) in faces_small]
            faces = [(x, y, w, h) for (x, y, w, h) in faces if min(w, h) >= 48]
            static["boxes"] = faces

        # LED feedback on Pi
        if self.use_picam and len(self.led_pins) > 0:
            try:
                import RPi.GPIO as GPIO
                GPIO.output(self.led_pins[0], GPIO.HIGH if static['boxes'] else GPIO.LOW)
            except Exception:
                pass

        for (x, y, w, h) in static["boxes"]:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame_bgr