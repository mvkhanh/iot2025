#!/usr/bin/env python3
import argparse
import threading, queue
import numpy as np
import cv2
import imagezmq

def receiver_thread(q: queue.Queue, bind: str, port: int, stop_event: threading.Event):
    """
    Luồng phụ (daemon): nhận JPEG qua ImageZMQ và đưa vào queue.
    Dùng poll(1000 ms) để có thể kiểm tra stop_event định kỳ (không bị block vô hạn).
    """
    hub = imagezmq.ImageHub(open_port=f"tcp://{bind}:{port}", REQ_REP=True)
    print(f"[SERVER] Listening on tcp://{bind}:{port}")
    try:
        while not stop_event.is_set():
            # chờ tối đa 1000ms để kiểm tra stop_event
            try:
                if hasattr(hub, "zmq_socket") and hub.zmq_socket.poll(1000):  # 1s
                    name, jpg_buffer = hub.recv_jpg()
                    hub.send_reply(b"OK")
                    # giữ frame mới nhất
                    if q.full():
                        try:
                            q.get_nowait()
                        except Exception:
                            pass
                    try:
                        q.put_nowait(jpg_buffer)
                    except Exception:
                        pass
                else:
                    # không có dữ liệu trong 1s, quay lại check stop_event
                    continue
            except Exception as e:
                continue
    finally:
        # dọn dẹp socket/context của ZMQ
        try:
            if hasattr(hub, "zmq_socket"):
                hub.zmq_socket.close(0)
        except Exception:
            pass
        try:
            if hasattr(hub, "zmq_context"):
                hub.zmq_context.term()
        except Exception:
            pass
        print("[SERVER] Receiver thread cleaned up.")

def main(args):
    q = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # start receiver as daemon
    t_recv = threading.Thread(
        target=receiver_thread,
        args=(q, args.bind, args.port, stop_event),
        daemon=True
    )
    t_recv.start()

    # main thread runs display (blocking)
    win = args.window
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[SERVER] Display running. Press 'q' to quit.")
    try:
        while not stop_event.is_set():
            try:
                jpg_buffer = q.get(timeout=1.0)
            except queue.Empty:
                # vẫn cần quét phím thoát
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            arr = np.frombuffer(jpg_buffer, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            cv2.imshow(win, frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[SERVER] KeyboardInterrupt.")
    finally:
        # yêu cầu receiver dừng & dọn dẹp UI
        stop_event.set()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            t_recv.join(timeout=1.0)
        except Exception:
            pass
        print("[SERVER] Clean exit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server: 1 main (display) + 1 daemon (receiver)")
    parser.add_argument("--bind", default="*", help='Địa chỉ bind (thường "*" hoặc "0.0.0.0")')
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--window", default="Face Stream")
    args = parser.parse_args()
    main(args)