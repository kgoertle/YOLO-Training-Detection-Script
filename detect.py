import argparse
import sys
from pathlib import Path
from datetime import datetime
import threading
import time
import math
import numpy as np
import cv2
import csv
from ultralytics import YOLO

BASE_DIR = Path(r"C:\Users\TheGo\Documents\YOLO")

# -------------
# Class colors
# -------------
CLASS_COLORS = {
    "M": (255, 200, 180), "F": (255, 192, 203), "Feeder": (144, 238, 144),
    "Main_Perch": (50, 50, 50), "Sky_Perch": (200, 200, 200),
    "Wooden_Perch": (60, 105, 165), "Nesting_Box": (69, 98, 99)
}

# ------------------------
# Output directory system
# ------------------------

def find_latest_best(base_path):
    base_path = Path(base_path)
    if not base_path.exists(): return None
    train_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("train (")]
    if not train_dirs: return None

    def parse_ts(folder_name):
        try: return datetime.strptime(folder_name.replace("train (", "").replace(")", ""), "%m-%d-%Y %H-%M-%S")
        except: return datetime.min

    latest_dir = max(train_dirs, key=lambda d: parse_ts(d.name))
    best_pt = latest_dir / "weights" / "best.pt"
    return best_pt if best_pt.exists() else None

def get_output_folder(weights_path, source_type, source_name, test_detect=False):
    train_folder = weights_path.parent.parent
    base_logs = BASE_DIR / ("logs/test-runs" if test_detect else "logs/runs")
    out_base = base_logs / train_folder.name / "recordings"
    out_dir = out_base / ("video-in" if source_type=="video" else source_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# ------------------------------------------
# Temporal smoothing parameters and logging
# ------------------------------------------

class BoxSmoother:
    def __init__(self, max_history=0, alpha=1.0, dist_thresh=None):
        self.max_history = max_history
        self.alpha = alpha
        self.dist_thresh = dist_thresh
        self.history = []

    def smooth(self, boxes):
        smoothed, new_history = [], []
        for box in boxes:
            x1, y1, x2, y2, cls = box
            matched = None
            if self.dist_thresh is not None:
                for hx1, hy1, hx2, hy2, hcls in self.history:
                    if cls != hcls: continue
                    if np.linalg.norm(np.array([(x1+x2)/2,(y1+y2)/2])-np.array([(hx1+hx2)/2,(hy1+hy2)/2])) < self.dist_thresh:
                        matched = (hx1, hy1, hx2, hy2, hcls); break
            if matched:
                hx1, hy1, hx2, hy2, _ = matched
                x1 = int(self.alpha*x1 + (1-self.alpha)*hx1)
                y1 = int(self.alpha*y1 + (1-self.alpha)*hy1)
                x2 = int(self.alpha*x2 + (1-self.alpha)*hx2)
                y2 = int(self.alpha*y2 + (1-self.alpha)*hy2)
            smoothed.append([x1, y1, x2, y2, cls])
            new_history.append([x1, y1, x2, y2, cls])
        self.history = new_history[-self.max_history:]
        return smoothed

def report_smoothing_params(args, user_set_flags, default_params):
    lab_adjusted, user_adjusted, ultralytics_default = [], [], []
    for param, value in [('smooth', args.smooth), ('dist_thresh', args.dist_thresh), ('max_history', args.max_history)]:
        if user_set_flags.get(param, False): user_adjusted.append(f".. adjusted {param} to {value}")
        elif getattr(args, param) != default_params[param]: lab_adjusted.append(f".. adjusted {param} to {value}")
        else: ultralytics_default.append(f".. returning {param} to {value}")
    if args.lab and lab_adjusted: print("[INFO] Lab mode enabled."); [print(l) for l in lab_adjusted]
    if user_adjusted: print("[INFO] User adjusted parameters."); [print(l) for l in user_adjusted]
    if ultralytics_default: print("[INFO] Ultralytics default parameters."); [print(l) for l in ultralytics_default]

# ---------------
# CSV generation
# ---------------

def update_csv_file(csv_dir, interactions, session_start, session_end):
    """Create or update CSV file using dynamic 'start to end' filename."""
    csv_file = csv_dir / f"{session_start.strftime('%d-%m %H-%M-%S')} to {session_end.strftime('%d-%m %H-%M-%S')}.csv"
    file_exists = csv_file.exists()
    with open(csv_file,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(["Bird","Object","Start Time","End Time","Duration (s)","Frames"])
        for (bname,oname), data in interactions.items():
            writer.writerow([
                bname, oname,
                datetime.fromtimestamp(data["start_time"]).strftime("%H:%M:%S"),
                datetime.fromtimestamp(data.get("end_time", time.time())).strftime("%H:%M:%S"),
                f"{data.get('duration',0):.2f}",
                data.get("frames",0)
            ])
    return csv_file

# ---------------------
# Detection setup
# ---------------------

def run_detection(model, source, source_type, test_detect=False, smoother=None, src_idx=0, total_sources=1):
    source_name = f"{source_type}{source}" if source_type in ["usb","picamera"] else str(source)
    out_path = get_output_folder(model.weights_path, source_type, source_name, test_detect)
    timestamp_str = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
    out_file = out_path / f"{timestamp_str}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # -- recording setup
    
    if source_type == "usb":
        cap = cv2.VideoCapture(int(source))
    elif source_type == "picamera":
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888',"size":(1280,720)}))
        cap.start()
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap or (source_type!="picamera" and not cap.isOpened()):
        print(f"[ERROR] Could not open {source_name}")
        return
    print(f"[INFO] Running detection for {source_name}:")

    # -- window setup
    
    DISPLAY_BASE_WIDTH, DISPLAY_BASE_HEIGHT = 1280, 720
    cols = math.ceil(math.sqrt(total_sources))
    rows = math.ceil(total_sources / cols)
    display_width = min(DISPLAY_BASE_WIDTH, 1920 // cols)
    display_height = min(DISPLAY_BASE_HEIGHT, 1080 // rows)
    row_idx = src_idx // cols
    col_idx = src_idx % cols
    x_pos = col_idx * display_width
    y_pos = row_idx * display_height

    try:
        src_label = (Path(str(source)).name if source_type == "video" else f"{source_type}{source}")
    except Exception:
        src_label = f"{source_type}{source}"
    window_name = f"YOLO Detection - {src_label}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.moveWindow(window_name, x_pos, y_pos)

    out_writer = cv2.VideoWriter(str(out_file), fourcc, 30, (1280,720))
    prev_time, fps_smooth, esc_pressed = time.time(), 0, False
    interaction_counter = 0
    if smoother is None:
        smoother = BoxSmoother()

    # -----------------------------
    # Interaction tracking setup
    # -----------------------------
    
    targets = ["Feeder","Main_Perch","Sky_Perch","Nesting_Box","Wooden_Perch"]
    interaction_base = BASE_DIR / ("logs/test-runs" if test_detect else "logs/runs") / model.weights_path.parent.parent.name / "interaction-metrics"
    interaction_dir = interaction_base / ("video-in" if source_type=="video" else source_name)
    interaction_dir.mkdir(parents=True, exist_ok=True)
    interactions = {}  # (bird_class, object_class) -> {"start_time": float, "frames": int, "active": bool, "duration": float}

    threshold = 5 if test_detect else 60
    save_interval = 60 if test_detect else 3600  # seconds
    next_save_time = time.time() + save_interval
    session_start = datetime.now()

    while True:
        if esc_pressed: break
        ret, frame = (cap.read() if source_type!="picamera" else (True, cap.capture_array()))
        if not ret or frame is None: break

        frame_resized = cv2.resize(frame, (1280, 720))
        results, obj_count, smoothed_boxes_list = model.predict(frame_resized, verbose=False, show=False), 0, []

        for r in results:
            if hasattr(r,'boxes') and r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                combined = [[int(x1),int(y1),int(x2),int(y2),int(cls),float(conf)]
                            for (x1,y1,x2,y2),cls,conf in zip(boxes,classes,confs)]
                smoothed_boxes = smoother.smooth([[x1,y1,x2,y2,cls] for x1,y1,x2,y2,cls,_ in combined])
                smoothed_boxes_list.extend(smoothed_boxes); obj_count += len(smoothed_boxes)

                for i,(x1,y1,x2,y2,cls) in enumerate(smoothed_boxes):
                    cname, conf = r.names[cls], combined[i][5]
                    color = CLASS_COLORS.get(cname,(0,255,0))
                    label = f"{cname} {conf:.1f}"
                    cv2.rectangle(frame_resized,(x1,y1),(x2,y2),color,2)
                    (w,h),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                    cv2.rectangle(frame_resized,(x1,y1-h-4),(x1+w,y1),color,-1)
                    cv2.putText(frame_resized,label,(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

        # -- interaction detection
        
        now = time.time()
        M_or_F = [b for b in smoothed_boxes_list if r.names[b[4]] in ["M","F"]]
        for bird in M_or_F:
            bx1,by1,bx2,by2,bcls=bird; bname=r.names[bcls]
            for obj in smoothed_boxes_list:
                ox1,oy1,ox2,oy2,ocls=obj; oname=r.names[ocls]
                if bname==oname or oname not in targets: continue
                ix1,iy1,ix2,iy2=max(bx1,ox1),max(by1,oy1),min(bx2,ox2),min(by2,oy2)
                key=(bname,oname)
                if ix2>ix1 and iy2>iy1:
                    if key not in interactions:
                        interactions[key]={"start_time":now,"frames":1,"active":False,"duration":0}
                    else:
                        interactions[key]["frames"]+=1
                        interactions[key]["duration"]=now-interactions[key]["start_time"]
                    if interactions[key]["frames"]==threshold:
                        interactions[key]["active"]=True; interaction_counter+=1
                else:
                    if key in interactions and interactions[key]["active"]:
                        interactions[key]["active"]=False
                    interactions.pop(key,None)

        # -- periodic CSV save
        
        if now >= next_save_time:
            session_end = datetime.now()
            update_csv_file(interaction_dir, interactions, session_start, session_end)
            next_save_time = now + save_interval

        # -- FPS and other overlay preferences
        
        curr_time=time.time(); fps=1/(curr_time-prev_time)
        fps_smooth=0.9*fps_smooth+0.1*fps; prev_time=curr_time
        h = frame_resized.shape[0]
        cv2.putText(frame_resized,f"FPS: {fps_smooth:.1f}",(10,h-60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        cv2.putText(frame_resized,f"Objects: {obj_count}",(10,h-35),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        cv2.putText(frame_resized,f"Interactions: {interaction_counter}",(10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        cv2.imshow(window_name, frame_resized)
        out_writer.write(frame_resized)
        if cv2.waitKey(1)&0xFF==27:
            esc_pressed=True; print(f"[ESC] Closing {source_name}...")

    # ------------
    # ESC cleanup 
    # ------------
    
    if source_type in ["usb","video"]:
        cap.release()
    elif source_type=="picamera":
        cap.stop()
    if out_writer:
        out_writer.release()
    cv2.destroyWindow(window_name)

    # -- final save of remaining interactions with dynamic filename
    
    session_end = datetime.now()
    final_csv = update_csv_file(interaction_dir, interactions, session_start, session_end)
    print(f"[SAVE] Final interaction metrics saved to: {final_csv}")
    print(f"[SAVE] Detection results saved to: {out_file}")
    print(f"[SAVE] Interaction metrics folder: {interaction_dir}")


# ----------------
# Argument parser
# ----------------

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Run YOLO detection using latest best.pt")
    parser.add_argument("--detect",action="store_true")
    parser.add_argument("--test-detect",action="store_true")
    parser.add_argument("--sources",nargs='+',required=True)
    parser.add_argument("--lab",action="store_true",help="Enable lab mode with bird tracking-friendly smoothing parameters")
    parser.add_argument("--smooth",type=float,default=1.0)
    parser.add_argument("--dist-thresh",type=float,default=None)
    parser.add_argument("--max-history",type=int,default=0)
    args=parser.parse_args()

    if args.detect and args.test_detect: print("ERROR: Choose either --detect or --test-detect, not both."); sys.exit(1)
    weights_path=find_latest_best(BASE_DIR/("runs/detect" if args.detect else "runs/test"))
    if not weights_path: print("ERROR: Could not find a valid best.pt"); sys.exit(1)

    print(f"[INFO] Loading shared YOLO model from {weights_path}")
    model=YOLO(str(weights_path)); model.weights_path=weights_path

    user_set_flags={'smooth':'--smooth' in sys.argv,'dist_thresh':'--dist-thresh' in sys.argv,'max_history':'--max-history' in sys.argv}
    default_params={'smooth':1.0,'dist_thresh':None,'max_history':0}
    if args.lab:
        if not user_set_flags['smooth']: args.smooth=0.6
        if not user_set_flags['dist_thresh']: args.dist_thresh=70
        if not user_set_flags['max_history']: args.max_history=4
    report_smoothing_params(args,user_set_flags,default_params)

    smoothers={}; threads=[]
    for idx, src in enumerate(args.sources):
        if src.lower().startswith("usb"): source_type="usb"; src_id=int(src[3:])
        elif src.lower().startswith("picamera"): source_type="picamera"; src_id=int(src[9:])
        else: source_type="video"; src_id=src
        if src_id not in smoothers: smoothers[src_id]=BoxSmoother(max_history=args.max_history,alpha=args.smooth,dist_thresh=args.dist_thresh)
        t=threading.Thread(target=run_detection,args=(model,src_id,source_type,args.test_detect,smoothers[src_id],idx,len(args.sources)))
        t.start(); threads.append(t)
    for t in threads: t.join()
