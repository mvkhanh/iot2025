#!/usr/bin/env python3
import argparse
import time
import cv2
import imagezmq, zmq
from db.face_db import FaceDB
from model.face_detector import HaarFaceDetector
from model.face_recognizer import LBPFaceRecognizer
from utils import enroll_from_camera, VideoSource
from worker.recognize_worker import RecogWorker
from worker.detect_worker import DetectWorker

def detection(args):
    sender = imagezmq.ImageSender(connect_to=f"tcp://{args.server}:{args.port}", REQ_REP=True)
    # cấu hình timeout cho REQ socket (ms)
    sock = sender.zmq_socket
    sock.setsockopt(zmq.LINGER, 0)           # không chờ khi close
    sock.setsockopt(zmq.RCVTIMEO, 2000)      # 2s đợi reply từ server
    sock.setsockopt(zmq.SNDTIMEO, 2000)      # 2s đợi gửi
    print(f"[CLIENT] Connecting to tcp://{args.server}:{args.port}")
    
    detector = HaarFaceDetector()
    cam = VideoSource(args.width, args.height, args.fps, use_picam=args.use_picam)
    recog_worker = DetectWorker(detector=detector, use_picam=args.use_picam, led_pins=args.led_pins,
                               thresh=args.thresh, margin=args.margin, detect_every_n=args.den, quality=args.quality)
    recog_worker.start()
    
    consecutive_fail = 0
    MAX_FAILS = 3  # quá 3 lần lỗi liên tiếp thì dừng
    
    try:
        while True:
            ok, frame_bgr = cam.read()
            if not ok:
                time.sleep(0.02); continue
            recog_worker.submit(frame_bgr)
            
            jpg = recog_worker.last_jpg
            if jpg is None:
                ok, tmp = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                jpg = tmp.tobytes() if ok else None
            if jpg is None:
                continue
            
            try:
                # sẽ raise ZMQError nếu server tắt / timeout
                _ = sender.send_jpg(args.name, jpg)
                consecutive_fail = 0  # reset khi gửi OK
            except Exception as e:
                consecutive_fail += 1
                print(f"[CLIENT] send_jpg failed ({consecutive_fail}/{MAX_FAILS}): {e}")
                if consecutive_fail >= MAX_FAILS:
                    print("[CLIENT] Server unreachable. Stopping client.")
                    break
                # backoff nhẹ rồi thử lại, hoặc bạn có thể recreate socket
                time.sleep(0.5)
                # Option (recreate clean):
                try:
                    sender.zmq_socket.close(0)
                    sender.zmq_context.term()
                except Exception:
                    pass
                sender = imagezmq.ImageSender(connect_to=f"tcp://{args.server}:{args.port}", REQ_REP=True)
                sock = sender.zmq_socket
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVTIMEO, 2000)
                sock.setsockopt(zmq.SNDTIMEO, 2000)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        try:
            recog_worker.stop()
        except Exception:
            pass
        # dọn ZMQ
        try:
            sender.zmq_socket.close(0)
        except Exception:
            pass
        try:
            sender.zmq_context.term()
        except Exception:
            pass

def recognition(args):
    sender = imagezmq.ImageSender(connect_to=f"tcp://{args.server}:{args.port}", REQ_REP=True)
    # cấu hình timeout cho REQ socket (ms)
    sock = sender.zmq_socket
    sock.setsockopt(zmq.LINGER, 0)           # không chờ khi close
    sock.setsockopt(zmq.RCVTIMEO, 2000)      # 2s đợi reply từ server
    sock.setsockopt(zmq.SNDTIMEO, 2000)      # 2s đợi gửi
    print(f"[CLIENT] Connecting to tcp://{args.server}:{args.port}")
    
    detector = HaarFaceDetector()
    recognizer = LBPFaceRecognizer()
    db = FaceDB()
    cam = VideoSource(args.width, args.height, args.fps, use_picam=args.use_picam)
    recog_worker = RecogWorker(detector=detector, recognizer=recognizer, db=db, use_picam=args.use_picam, led_pins=args.led_pins,
                               thresh=args.thresh, margin=args.margin, detect_every_n=args.den, quality=args.quality)
    recog_worker.start()
    
    consecutive_fail = 0
    MAX_FAILS = 3  # quá 3 lần lỗi liên tiếp thì dừng
    
    try:
        while True:
            ok, frame_bgr = cam.read()
            if not ok:
                time.sleep(0.02); continue
            recog_worker.submit(frame_bgr)
            
            jpg = recog_worker.last_jpg
            if jpg is None:
                ok, tmp = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                jpg = tmp.tobytes() if ok else None
            if jpg is None:
                continue
            
            try:
                # sẽ raise ZMQError nếu server tắt / timeout
                _ = sender.send_jpg(args.name, jpg)
                consecutive_fail = 0  # reset khi gửi OK
            except Exception as e:
                consecutive_fail += 1
                print(f"[CLIENT] send_jpg failed ({consecutive_fail}/{MAX_FAILS}): {e}")
                if consecutive_fail >= MAX_FAILS:
                    print("[CLIENT] Server unreachable. Stopping client.")
                    break
                # backoff nhẹ rồi thử lại, hoặc bạn có thể recreate socket
                time.sleep(0.5)
                # Option (recreate clean):
                try:
                    sender.zmq_socket.close(0)
                    sender.zmq_context.term()
                except Exception:
                    pass
                sender = imagezmq.ImageSender(connect_to=f"tcp://{args.server}:{args.port}", REQ_REP=True)
                sock = sender.zmq_socket
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVTIMEO, 2000)
                sock.setsockopt(zmq.SNDTIMEO, 2000)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        try:
            recog_worker.stop()
        except Exception:
            pass
        # dọn ZMQ
        try:
            sender.zmq_socket.close(0)
        except Exception:
            pass
        try:
            sender.zmq_context.term()
        except Exception:
            pass
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Raspberry Client")
    
    parser.add_argument("--server", help="IP/host của PC server")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--name", default="pi")
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

    if args.mode == 'recognition':
        if args.enroll_from_camera:
            enroll_from_camera(args.enroll_from_camera, args.num, args.width, args.height, args.fps, args.use_picam)
        elif args.server:
            recognition(args)
        else:
            print('Chưa nhập IP server!')
            
    elif args.mode == 'detection':
        if args.server:
            detection(args)
        else:
            print('Chưa nhập IP server!')
    if args.use_picam:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        for led in args.led_pins:
            GPIO.setup(led, GPIO.OUT) #led
            GPIO.output(led, GPIO.LOW)