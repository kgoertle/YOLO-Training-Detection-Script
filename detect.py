import argparse, sys, threading, time, csv, platform
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
stop_event = threading.Event()  # global stop event

CLASS_COLORS = {
    "M": (255, 200, 180), "F": (255, 192, 203), "Feeder": (144, 238, 144),
    "Main_Perch": (50, 50, 50), "Sky_Perch": (200, 200, 200),
    "Wooden_Perch": (60, 105, 165), "Nesting_Box": (69, 98, 99)
}

# ----- INPUT & OUTPUT DIRECTORIES ------
def find_latest_best(base_path): # defines where best.pt is to be used
    base_path = Path(base_path)
    dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("train (")]
    if not dirs: return None
    def parse_ts(name):
        try: return datetime.strptime(name.replace("train (","").replace(")",""), "%m-%d-%Y %H-%M-%S") # looks for timestamp
        except: return datetime.min
    latest = max(dirs, key=lambda d: parse_ts(d.name))
    pt = latest / "weights" / "best.pt"
    return pt if pt.exists() else None

def get_output_folder(weights_path, source_type, source_name, test_detect=False):
    train_folder = weights_path.parent.parent
    out_dir = BASE_DIR / ("logs/test-runs" if test_detect else "logs/runs") / train_folder.name / "recordings"
    out_dir = out_dir / ("video-in" if source_type=="video" else source_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# ----- BOX SMOOTHING -----
class BoxSmoother:
    def __init__(self, max_history=0, alpha=1.0, dist_thresh=None):
        self.max_history, self.alpha, self.dist_thresh = max_history, alpha, dist_thresh
        self.history = []

    def smooth(self, boxes):
        new_history, smoothed = [], []
        for box in boxes:
            x1,y1,x2,y2,cls = box
            matched = next(((hx1,hy1,hx2,hy2,hcls) for hx1,hy1,hx2,hy2,hcls in self.history
                            if cls==hcls and (self.dist_thresh is None or 
                            np.linalg.norm(np.array([(x1+x2)/2,(y1+y2)/2])-np.array([(hx1+hx2)/2,(hy1+hy2)/2]))<self.dist_thresh)), None)
            if matched:
                hx1,hy1,hx2,hy2,_ = matched
                x1 = int(self.alpha*x1 + (1-self.alpha)*hx1)
                y1 = int(self.alpha*y1 + (1-self.alpha)*hy1)
                x2 = int(self.alpha*x2 + (1-self.alpha)*hx2)
                y2 = int(self.alpha*y2 + (1-self.alpha)*hy2)
            smoothed.append([x1,y1,x2,y2,cls])
            new_history.append([x1,y1,x2,y2,cls])
        self.history = new_history[-self.max_history:]
        return smoothed

def report_smoothing_params(args, user_set_flags, default_params):
    for param, value in [('smooth', args.smooth), ('dist_thresh', args.dist_thresh), ('max_history', args.max_history)]:
        msg = f".. adjusted {param} to {value}" if user_set_flags.get(param, False) else f".. returning {param} to {value}"
        if args.lab: print("[INFO] Lab mode enabled." if param=="smooth" else msg)
        else: print("[INFO] User adjusted parameters." if user_set_flags.get(param, False) else "[INFO] Ultralytics default parameters."); print(msg)

# ----- CSV UPDATE -----
def update_csv_file(csv_dir, interactions, start, end):
    file = csv_dir / f"{start.strftime('%d-%m %H-%M-%S')} to {end.strftime('%d-%m %H-%M-%S')}.csv"
    with open(file,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(["Bird","Object","Start Time","End Time","Duration (s)","Frames"])
        for (bname,oname), data in interactions.items():
            w.writerow([
                bname, oname,
                datetime.fromtimestamp(data["start_time"]).strftime("%H:%M:%S"),
                datetime.fromtimestamp(data.get("end_time", time.time())).strftime("%H:%M:%S"),
                f"{data.get('duration',0):.2f}", data.get("frames",0)
            ])
    return file

# ----- DETECTION LOOP -----
def run_detection(model, source, source_type, test_detect=False, smoother=None): # main argument definitions
    source_name = f"{source_type}{source}" if source_type in ["usb","picamera"] else str(source)
    out_path = get_output_folder(model.weights_path, source_type, source_name, test_detect)
    out_file = out_path / f"{datetime.now().strftime('%m-%d-%Y %H-%M-%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    IS_MAC, IS_LINUX, IS_WINDOWS = platform.system() == "Darwin", platform.system() == "Linux", platform.system() == "Windows" # works to ensure whichever system is using proper video encoder
    if source_type == "usb":
        cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_V4L2 if IS_LINUX else 0)
    elif source_type == "picamera":
        if not IS_LINUX: raise RuntimeError("[ERROR] Pi Camera only supported on Linux.")
        from picamera2 import Picamera2
        cap = Picamera2(); cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (1280,720)})); cap.start()
    else:
        cap = cv2.VideoCapture(str(source))
    if not cap or (source_type!="picamera" and not cap.isOpened()): print(f"[ERROR] Could not open {source_name}!"); return

    print(f"[INFO] Running detection for {source_name}:")
    out_writer = cv2.VideoWriter(str(out_file), fourcc, 30, (1280,720))
    prev_time, fps_smooth, frame_count, interaction_counter = time.time(), 0, 0, 0
    smoother = smoother or BoxSmoother()

    targets = ["Feeder","Main_Perch","Sky_Perch","Nesting_Box","Wooden_Perch"] # defines interaction targets
    interaction_dir = BASE_DIR / ("logs/test-runs" if test_detect else "logs/runs") / model.weights_path.parent.parent.name / "interaction-metrics" / ("video-in" if source_type=="video" else source_name) # output directories for proper type and source
    interaction_dir.mkdir(parents=True, exist_ok=True)
    interactions, threshold, save_interval, next_save_time = {}, 5 if test_detect else 60, 60 if test_detect else 3600, time.time() + (60 if test_detect else 3600) # determines CVS saving interval for detect argument
    session_start = datetime.now()

    try:
        while not stop_event.is_set(): 
            ret, frame = (cap.read() if source_type!="picamera" else (True, cap.capture_array())) 
            if not ret or frame is None: break
            frame_resized = cv2.resize(frame, (1280,720))
            results = model.predict(frame_resized, verbose=False, show=False)
            smoothed_boxes_list, obj_count = [], 0

            for r in results: # smoother function with color association and style choice
                if hasattr(r, 'boxes') and r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    combined = [[int(x1),int(y1),int(x2),int(y2),int(cls),float(conf)]
                                for (x1,y1,x2,y2),cls,conf in zip(boxes,classes,confs)]
                    smoothed_boxes = smoother.smooth([[x1,y1,x2,y2,cls] for x1,y1,x2,y2,cls,_ in combined])
                    smoothed_boxes_list.extend(smoothed_boxes)
                    obj_count += len(smoothed_boxes)
                    for i,(x1,y1,x2,y2,cls) in enumerate(smoothed_boxes):
                        cname, conf = r.names[cls], combined[i][5]
                        color = CLASS_COLORS.get(cname,(0,255,0))
                        label=f"{cname} {conf:.1f}"
                        cv2.rectangle(frame_resized,(x1,y1),(x2,y2),color,2)
                        (w,h),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                        cv2.rectangle(frame_resized,(x1,y1-h-4),(x1+w,y1),color,-1)
                        cv2.putText(frame_resized,label,(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

            # -- main interaction detection -- 
            now = time.time()
            M_or_F = [b for b in smoothed_boxes_list if r.names[b[4]] in ["M","F"]]
            for bx1,by1,bx2,by2,bcls in M_or_F:
                bname = r.names[bcls]
                for ox1,oy1,ox2,oy2,ocls in smoothed_boxes_list:
                    oname = r.names[ocls]
                    if bname==oname or oname not in targets: continue
                    ix1,iy1,ix2,iy2 = max(bx1,ox1), max(by1,oy1), min(bx2,ox2), min(by2,oy2)
                    key=(bname,oname)
                    if ix2>ix1 and iy2>iy1:
                        if key not in interactions: interactions[key]={"start_time":now,"frames":1,"active":False,"duration":0}
                        else: interactions[key]["frames"]+=1; interactions[key]["duration"]=now-interactions[key]["start_time"]
                        if interactions[key]["frames"]==threshold: interactions[key]["active"]=True; interaction_counter+=1
                    else:
                        if key in interactions and interactions[key]["active"]: interactions[key]["active"]=False
                        interactions.pop(key,None)

            # -- periodic CSV save --
            if now>=next_save_time:
                update_csv_file(interaction_dir, interactions, session_start, datetime.now())
                next_save_time = now+save_interval

            # -- FPS & display -- 
            fps_smooth = 0.9*fps_smooth + 0.1*(1/(time.time()-prev_time)); prev_time=time.time()
            frame_count+=1
            print(f"\rFrames: {frame_count} | Objects: {obj_count} | Interactions: {interaction_counter} | FPS: {fps_smooth:.1f}", end="")
            out_writer.write(frame_resized)

    finally: # -- safe quit with logs --
        if source_type in ["usb","video"]: cap.release()
        elif source_type=="picamera": cap.stop()
        out_writer.release()
        final_csv = update_csv_file(interaction_dir, interactions, session_start, datetime.now())
        print(f"\n[SAVE] Final interaction metrics saved to: {final_csv}")
        print(f"[SAVE] Detection results saved to: {out_file}")
        print(f"[SAVE] Interaction metrics folder: {interaction_dir}")

# ------ ARGUMENT PARSER -------
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run YOLO detection using latest best.pt")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--test-detect", action="store_true")
    parser.add_argument("--sources", nargs='+', required=True)
    parser.add_argument("--lab", action="store_true")
    parser.add_argument("--smooth", type=float, default=1.0)
    parser.add_argument("--dist-thresh", type=float, default=None)
    parser.add_argument("--max-history", type=int, default=0)
    args = parser.parse_args()

    if args.detect and args.test_detect: print("[ERROR] Choose either --detect or --test-detect, not both."); sys.exit(1)
    weights_path = find_latest_best(BASE_DIR/("runs/detect" if args.detect else "runs/test"))
    if not weights_path: print("[ERROR] Could not find a valid best.pt"); sys.exit(1)

    print(f"[INFO] Loading shared YOLO model from {weights_path}.")
    model = YOLO(str(weights_path)); model.weights_path = weights_path

    user_set_flags = {'smooth':'--smooth' in sys.argv,'dist_thresh':'--dist-thresh' in sys.argv,'max_history':'--max-history' in sys.argv} # this allows us to parse our own temporal smoothing parameters in terminal
    if args.lab: # set up for --lab report for parameter options
        if not user_set_flags['smooth']: args.smooth=0.6
        if not user_set_flags['dist_thresh']: args.dist_thresh=70
        if not user_set_flags['max_history']: args.max_history=4
    report_smoothing_params(args,user_set_flags,{'smooth':1.0,'dist_thresh':None,'max_history':0})

    smoothers, threads = {}, [] # imbeds smoothing parameters with run_detection parses
    for src in args.sources:
        if src.lower().startswith("usb"): source_type="usb"; src_id=int(src[3:])
        elif src.lower().startswith("picamera"): source_type="picamera"; src_id=int(src[9:])
        else: source_type="video"; src_id=src
        if src_id not in smoothers: smoothers[src_id] = BoxSmoother(max_history=args.max_history, alpha=args.smooth, dist_thresh=args.dist_thresh)
        t = threading.Thread(target=run_detection, args=(model, src_id, source_type, args.test_detect, smoothers[src_id]))
        t.start(); threads.append(t)

    try: [t.join() for t in threads] # closes safely and saves metrics
    except KeyboardInterrupt:
        print("\n[EXIT] Stop signal received. Terminating threads...")
        stop_event.set(); [t.join() for t in threads]
    print("[EXIT] All detection threads safely terminated.")
