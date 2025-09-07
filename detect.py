# ====== MULTITHREADED YOLO DETECTION SCRIPT WITH IMPROVED LOGGING AND DASHBOARD ======
import argparse, sys, threading, time, platform, os
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
from pymediainfo import MediaInfo

BASE_DIR = Path(__file__).resolve().parent
stop_event = threading.Event()
print_lock = threading.Lock()


# ----- REPORT SMOOTHING PARAMS -----
def report_smoothing_params(args, user_set_flags, default_params):
    for param in ['smooth', 'dist_thresh', 'max_history']:
        value = getattr(args, param)
        user_set = user_set_flags.get(param, False)
        if user_set:
            with print_lock:
                print(f"[INFO] {param} set by user to {value}")
        else:
            with print_lock:
                print(f"[INFO] {param} using {value}")


# ----- INPUT & OUTPUT DIRECTORIES -----
def find_latest_best(base_path):
    base_path = Path(base_path)
    if not base_path.exists():
        return None
    dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not dirs:
        return None

    def parse_ts(name):
        try:
            return datetime.strptime(name, "%m-%d-%Y_%H-%M-%S")
        except:
            return datetime.min

    latest = max(dirs, key=lambda d: parse_ts(d.name))
    pt = latest / "weights" / "best.pt"
    return pt if pt.exists() else None


def get_output_folder(weights_path, source_type, source_name, test_detect=False):
    train_folder = weights_path.parent.parent
    out_dir = BASE_DIR / ("logs/test" if test_detect else "logs/main") / train_folder.name / "recordings"
    out_dir = out_dir / ("video-in" if source_type == "video" else source_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ----- VIDEO ROTATION -----
def get_rotation_angle(video_path):
    try:
        media_info = MediaInfo.parse(str(video_path))
        for track in media_info.tracks:
            if track.track_type == "Video":
                rotation = getattr(track, "rotation", 0)
                if rotation:
                    return int(float(rotation))
    except Exception as e:
        with print_lock:
            print(f"[WARN] Could not read rotation from {video_path}: {e}")
    return 0


def rotate_frame(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ----- BOX SMOOTHING -----
class BoxSmoother:
    def __init__(self, max_history=5, alpha=0.6, dist_thresh=None):
        self.max_history = max_history
        self.alpha = alpha
        self.dist_thresh = dist_thresh
        self.history = []

    @staticmethod
    def smooth_angle(prev, new, alpha=0.6):
        diff = ((new - prev + 180) % 360) - 180
        return prev + alpha * diff

    def smooth(self, boxes):
        smoothed = []
        new_history = []
        for box in boxes:
            cx, cy, w, h, angle, cls = box
            matched = None
            for hx, hy, hw, hh, hangle, hcls in self.history:
                if cls != hcls:
                    continue
                if self.dist_thresh is None or np.linalg.norm([cx - hx, cy - hy]) < self.dist_thresh:
                    matched = (hx, hy, hw, hh, hangle, hcls)
                    break
            if matched:
                hx, hy, hw, hh, hangle, _ = matched
                cx = self.alpha * cx + (1 - self.alpha) * hx
                cy = self.alpha * cy + (1 - self.alpha) * hy
                w = self.alpha * w + (1 - self.alpha) * hw
                h = self.alpha * h + (1 - self.alpha) * hh
                angle = self.smooth_angle(hangle, angle, self.alpha)
                angle = ((angle + 180) % 360) - 180
            smoothed.append([cx, cy, w, h, angle, cls])
            new_history.append([cx, cy, w, h, angle, cls])
        self.history = new_history[-self.max_history:]
        return smoothed


# ----- DASHBOARD HANDLER -----
class Dashboard:
    def __init__(self, total_sources):
        self.total_sources = total_sources
        self.lock = threading.Lock()
        self.lines = [""] * total_sources  # current displayed text
        term_height = os.get_terminal_size().lines
        self.start_line = max(1, term_height - total_sources + 1)

    def update(self, line_number, text):
        """
        line_number: 1-indexed thread line number
        text: string to display for this thread
        """
        with self.lock:
            if self.lines[line_number - 1] != text:
                self.lines[line_number - 1] = text
                print(f"\033[{self.start_line + line_number - 1};0H\033[K{text}", end="", flush=True)


# ----- DETECTION LOOP -----
def run_detection(model, source, source_type, line_number, total_sources, dashboard, test_detect=False, smoother=None):
    source_name = f"{source_type}{source}" if source_type in ["usb", "picamera"] else str(source)
    source_label = f"[{source_name}]"

    out_path = get_output_folder(model.weights_path, source_type, source_name, test_detect)
    out_file = out_path / f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    IS_MAC, IS_LINUX = platform.system() == "Darwin", platform.system() == "Linux"

    if source_type == "usb":
        cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_V4L2 if IS_LINUX else 0)
    elif source_type == "picamera":
        if not IS_LINUX:
            print("[ERROR] Pi Camera only supported on Linux.")
            return
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (1280, 720)}))
        cap.start()
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap or (source_type != "picamera" and not cap.isOpened()):
        print(f"[ERROR] Could not open {source_name}!")
        return

    rotation_angle = get_rotation_angle(source) if source_type == "video" else 0
    ret, frame = cap.read() if source_type != "picamera" else (True, cap.capture_array())
    if not ret or frame is None:
        print(f"[ERROR] Could not read a frame from {source_name}")
        return

    if source_type == "picamera":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if rotation_angle in [90, 180, 270]:
        frame = rotate_frame(frame, rotation_angle)
    elif frame.shape[0] > frame.shape[1]:
        frame = rotate_frame(frame, 90)

    height, width = frame.shape[:2]
    out_writer = cv2.VideoWriter(str(out_file), fourcc, cap.get(cv2.CAP_PROP_FPS) or 30, (width, height))

    smoother = smoother or BoxSmoother()
    frame_count, fps_smooth, prev_time = 0, 0, time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if source_type == "video" else None
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30 if source_type == "video" else None

    try:
        while not stop_event.is_set():
            ret, frame = cap.read() if source_type != "picamera" else (True, cap.capture_array())
            if not ret or frame is None:
                break
            if source_type == "picamera":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if rotation_angle in [90, 180, 270]:
                frame = rotate_frame(frame, rotation_angle)
            elif frame.shape[0] > frame.shape[1]:
                frame = rotate_frame(frame, 90)

            draw_frame = frame.copy()
            results = model.predict(draw_frame, verbose=False, show=False)
            draw_frame = results[0].plot() if results else draw_frame

            smoothed_boxes_list = []
            if results and hasattr(results[0], 'obb') and results[0].obb is not None:
                boxes = results[0].obb.xywhr.cpu().numpy()
                classes = results[0].obb.cls.cpu().numpy()
                smoothed_boxes = smoother.smooth([
                    [cx, cy, w, h, float(angle), int(cls)]
                    for cx, cy, w, h, angle, cls in zip(
                        boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], classes
                    )
                ])
                smoothed_boxes_list.extend(smoothed_boxes)

            names = results[0].names if results else {}

            fps_smooth = 0.9 * fps_smooth + 0.1 * (1 / (time.time() - prev_time + 1e-6))
            prev_time = time.time()
            frame_count += 1

            males = sum(1 for b in smoothed_boxes_list if names.get(b[5]) == "M")
            females = sum(1 for b in smoothed_boxes_list if names.get(b[5]) == "F")
            other_objects = sum(1 for b in smoothed_boxes_list if names.get(b[5]) not in ["M", "F"])

            if source_type == "video":
                elapsed_sec = frame_count / fps_video
                total_sec = total_frames / fps_video if total_frames else 0
                time_info = f"{int(elapsed_sec // 60):02d}:{int(elapsed_sec % 60):02d}/{int(total_sec // 60):02d}:{int(total_sec % 60):02d}"
            else:
                elapsed_sec = int(time.time() - prev_time)
                h, m = divmod(elapsed_sec // 60, 60)
                s = elapsed_sec % 60
                time_info = f"{h:02d}:{m:02d}:{s:02d}"

            dashboard.update(
                line_number,
                f"[{source_name}] Frames:{frame_count} | FPS:{fps_smooth:.1f} | "
                f"Males:{males} | Females:{females} | Objects:{other_objects} | Time:{time_info}"
            )
            out_writer.write(draw_frame)

    finally:
        if source_type in ["usb", "video"]:
            cap.release()
        elif source_type == "picamera":
            cap.stop()
        out_writer.release()
        with print_lock:
            print(f"\033[{dashboard.start_line + dashboard.total_sources};0H[SAVE] Detection results saved to: {out_file}")


# ------ ARGUMENT PARSER -------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO detection using latest best.pt")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--test-detect", action="store_true")
    parser.add_argument("--sources", nargs='+', required=True)
    parser.add_argument("--lab", action="store_true")
    parser.add_argument("--smooth", type=float, default=1.0)
    parser.add_argument("--dist-thresh", type=float, default=None)
    parser.add_argument("--max-history", type=int, default=0)
    args = parser.parse_args()

    if args.detect and args.test_detect:
        print("[ERROR] Please decide between --detect or --test-detect, not both.")
        sys.exit(1)

    runs_dir = BASE_DIR / ("runs/main" if args.detect else "runs/test")
    weights_path = find_latest_best(runs_dir)
    if not weights_path:
        print(f"[ERROR] Could not find a valid best.pt in {runs_dir}")
        sys.exit(1)

    print(f"[INFO] Loading model from {weights_path}.")
    model = YOLO(str(weights_path))
    model.weights_path = weights_path

    user_set_flags = {
        'smooth': '--smooth' in sys.argv,
        'dist_thresh': '--dist-thresh' in sys.argv,
        'max_history': '--max-history' in sys.argv
    }

    if args.lab:
        if not user_set_flags['smooth']:
            args.smooth = 0.6
        if not user_set_flags['dist_thresh']:
            args.dist_thresh = 70
        if not user_set_flags['max_history']:
            args.max_history = 4

    report_smoothing_params(args, user_set_flags, {'smooth': 1.0, 'dist_thresh': None, 'max_history': 0})

    smoothers, threads = {}, []
    total_sources = len(args.sources)
    dashboard = Dashboard(total_sources)

    for i, src in enumerate(args.sources, start=1):
        try:
            if src.lower().startswith("usb"):
                source_type = "usb"; src_id = int(src[3:])
            elif src.lower().startswith("picamera"):
                source_type = "picamera"; src_id = int(src[9:])
            else:
                source_type = "video"; src_id = src
        except ValueError:
            print(f"[ERROR] Invalid source ID format: {src}")
            continue

        if src_id not in smoothers:
            smoothers[src_id] = BoxSmoother(
                max_history=args.max_history,
                alpha=args.smooth,
                dist_thresh=args.dist_thresh
            )

        t = threading.Thread(
            target=run_detection,
            args=(model, src_id, source_type, i, total_sources, dashboard, args.test_detect, smoothers[src_id])
        )
        t.start()
        threads.append(t)

    try:
        [t.join() for t in threads]
    except KeyboardInterrupt:
        print("\n[EXIT] Stop signal received. Terminating threads...")
        stop_event.set()
        [t.join() for t in threads]
        print("[EXIT] All detection threads safely terminated.")
