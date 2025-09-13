#!/usr/bin/env python3
import argparse
import time
import cv2
import threading
from db.face_db import FaceDB
from model.face_detector import HaarFaceDetector
from model.face_recognizer import LBPFaceRecognizer
from utils import enroll_from_camera, VideoSource
from worker.detect_worker import DetectWorker
from worker.recognize_worker import RecogWorker

class CaptureWorker(threading.Thread):
    def __init__(self, cam: VideoSource, detect_worker: DetectWorker = None):
        super().__init__(daemon=True)
        self._stop = False
        self.detect_worker = detect_worker
        self.cam = cam
        
    def run(self):
        try:
            while not self._stop:
                ok, frame_bgr = self.cam.read()
                if not ok:
                    time.sleep(0.02); continue
                self.detect_worker.submit(frame_bgr)
        finally:
            self.cam.release()
            
    def stop(self):
        self._stop = True

def main(args, detect_worker: DetectWorker, capture_worker: CaptureWorker):
    capture_worker.start()
    detect_worker.start()

    cv2.namedWindow(args.mode, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.mode, width=640, height=480)
    try:
        while True:
            frame = detect_worker.poll()
            if frame is not None:
                cv2.imshow(args.mode, frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Raspberry Client")
    
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--den", type=int, default=3, help="detect_every_n")
    parser.add_argument("--use-cam", dest='use_picam', action="store_false", help="Dùng cam laptop trong trường hợp không có pi", default=True)
    parser.add_argument("--fps", type=int, default=15, help="FPS khi dùng cam laptop")
    parser.add_argument('--led-pins', type=lambda s: list(map(lambda x: int(x.strip()), s.split(','))), help='Ví dụ: 1,2,3', default=[21, 20, 16])
    
    sub = parser.add_subparsers(dest="mode", required=True)
    
    pc = sub.add_parser("recognition", help="Chạy recognition")
    pc.add_argument("--thresh", type=float, default=0.7, help="Chi-square threshold for LBP (try 0.55..0.70)")
    pc.add_argument("--margin", type=float, default=0.02)
    pc.add_argument("--enroll-from-camera", type=str, default=None, help="Tên người để enroll từ camera.")
    pc.add_argument("--num", type=int, default=15, help="Số mẫu khi enroll từ camera")
    
    pd = sub.add_parser("detection", help="Chạy detection")
    args = parser.parse_args()
    detector = HaarFaceDetector()
    cam = VideoSource(args.width, args.height, args.fps, use_picam=args.use_picam)
    
    if args.use_picam:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        for led in args.led_pins:
            GPIO.setup(led, GPIO.OUT) #led
            GPIO.output(led, GPIO.LOW)
            
    if args.mode == 'recognition':
        recognizer = LBPFaceRecognizer()  # kept for enroll flow
        db = FaceDB()                     # kept for enroll flow
        if args.enroll_from_camera:
            enroll_from_camera(name=args.enroll_from_camera, cam=cam, detector=HaarFaceDetector(),
                               recognizer=recognizer, db=db, num=args.num)
        else:
            recog_worker = RecogWorker(detector_cls=HaarFaceDetector, recognizer_cls=LBPFaceRecognizer,
                                       use_picam=args.use_picam, led_pins=args.led_pins,
                                       thresh=args.thresh, margin=args.margin,
                                       detect_every_n=args.den)
            capture_worker = CaptureWorker(cam, detect_worker=recog_worker)
            main(args, recog_worker, capture_worker)

    elif args.mode == 'detection':
        detect_worker = DetectWorker(detector_cls=HaarFaceDetector, use_picam=args.use_picam, led_pins=args.led_pins,
                                     detect_every_n=args.den)
        capture_worker = CaptureWorker(cam, detect_worker=detect_worker)
        main(args, detect_worker, capture_worker)
