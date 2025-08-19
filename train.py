import time
time.sleep(10000000000)
while True:
    time.sleep(4)
from ultralytics import YOLO

# Load the pre-trained YOLOv8 pose model
model = YOLO('yolo11n-pose.pt')
model.model.kpt_flip = [
    24, 25, 26, 27, 28, 29,   # 0–5 ↔ 26–31
    22, 23, 21,
    17, 18, 19, 20,  # 6–12 ↔ 24–19
    13, 14, 15, 16,   # center keypoints stay
    9, 10, 11, 12,
    8, 6, 7,   # 19–25 ↔ 9–7
    0, 1, 2, 3, 4, 5,          # 26–31 ↔ 0–5
    31, 30
]

# Train the model
model.train(
    data=f'data.yaml',
    task='pose',
    mode='train',
    epochs=500,
    imgsz=640,
    batch=16,
    mosaic=0.0,
    plots=True
)
