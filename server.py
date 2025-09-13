#!/usr/bin/env python3
import argparse
import time
import cv2
from db.face_db import FaceDB
from model.face_detector import HaarFaceDetector
from model.face_recognizer import LBPFaceRecognizer
from utils import enroll_from_camera, VideoSource
from worker.detect_worker import DetectWorker
from worker.recognize_worker import RecogWorker

def main(args, worker: DetectWorker, cam:VideoSource):
    worker.start()
    try:
        while True:
            ok, frame_bgr = cam.read()
            if not ok:
                time.sleep(0.02); continue
            worker.submit(frame_bgr)
            
            jpg = worker.last_jpg
            if jpg:
                cv2.imshow(args.mode, jpg)
            if cv2.waitKey(10) == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Raspberry Client")
    
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--den", type=int, default=3, help="detect_every_n")
    parser.add_argument("--use-cam", dest='use_picam', action="store_false", help="Dùng cam laptop trong trường hợp không có pi", default=True)
    parser.add_argument("--fps", type=int, default=15, help="FPS khi dùng cam laptop")
    parser.add_argument('--led-pins', type=lambda s: list(map(lambda x: int(x.strip()), s.split(','))), help='Ví dụ: 1,2,3')
    
    sub = parser.add_subparsers(dest="mode", required=True)
    
    pc = sub.add_parser("recognition", help="Chạy recognition")
    pc.add_argument("--thresh", type=float, default=0.6, help="Chi-square threshold for LBP (try 0.55..0.70)")
    pc.add_argument("--margin", type=float, default=0.02)
    pc.add_argument("--enroll-from-camera", type=str, default=None, help="Tên người để enroll từ camera.")
    pc.add_argument("--num", type=int, default=15, help="Số mẫu khi enroll từ camera")
    
    pd = sub.add_parser("detection", help="Chạy detection")
    args = parser.parse_args()
    detector = HaarFaceDetector()
    cam = VideoSource(args.width, args.height, args.fps, use_picam=args.use_picam)
    if args.mode == 'recognition':
        if args.enroll_from_camera:
            enroll_from_camera(args.enroll_from_camera, args.num, args.width, args.height, args.fps, args.use_picam)
        else:
            recognizer = LBPFaceRecognizer()
            db = FaceDB()
            recog_worker = RecogWorker(detector=detector, recognizer=recognizer, db=db, use_picam=args.use_picam, led_pins=args.led_pins,
                               thresh=args.thresh, margin=args.margin, detect_every_n=args.den, quality=args.quality)
            main(args, recog_worker)

    elif args.mode == 'detection':
        detect_worker = DetectWorker(detector=detector, use_picam=args.use_picam, led_pins=args.led_pins,
                               thresh=args.thresh, margin=args.margin, detect_every_n=args.den, quality=args.quality)
        main(args, detect_worker)

    if args.use_picam:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        for led in args.led_pins:
            GPIO.setup(led, GPIO.OUT) #led
            GPIO.output(led, GPIO.LOW)