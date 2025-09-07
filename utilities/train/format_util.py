import os
from pathlib import Path
import tensorflow as tf
from ultralytics import YOLO

def force_cpu():
    tf.config.set_visible_devices([], 'GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def export_tflite(model: YOLO, data_yaml: Path, imgsz=640, float32=False, float16=False):
    force_cpu() 

    try:
        if float32:
            print("[INFO] Exporting float32 TFLite model on CPU...")
            model.export(
                format="tflite",
                int8=False,
                half=False,
                imgsz=imgsz,
                batch=1,
                data=str(data_yaml),
                device="cpu"
            )

        if float16:
            print("[INFO] Exporting float16 TFLite model on CPU...")
            model.export(
                format="tflite",
                int8=False,
                half=True,
                imgsz=imgsz,
                batch=1,
                data=str(data_yaml),
                device="cpu"
            )

        if not (float32 or float16):
            print("[INFO] No TFLite export requested.")
        else:
            print("[INFO] TFLite export(s) complete.")

    except Exception as e:
        print(f"[ERROR] TFLite export failed: {e}")
