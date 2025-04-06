import cv2
from ultralytics import YOLO





# Load model YOLOv8 Instance Segmentation

model = YOLO("yolov8n-seg.pt")



def detect_rail_lane(image_path):

    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""

    results = model(image_path, show=True)

    results[0].save("lane_detection_result.jpg")



# Contoh penggunaan

detect_rail_lane("rel kereta api.jpg")

from ultralytics import YOLO

import cv2



# Load model YOLOv8 Instance Segmentation

model = YOLO("yolov8n-seg.pt")



def detect_rail_lane(image_path):

    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""

    results = model(image_path, show=True)

    results[0].save("lane_detection_result.jpg")



# Contoh penggunaan

detect_rail_lane("rel kereta api.jpg")

