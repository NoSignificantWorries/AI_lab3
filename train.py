from ultralytics import YOLO

# model = YOLO('yolov8s.pt')
model = YOLO('runs/detect/train2/weights/best.pt')

model.train(data='data/dataset/data.yaml', epochs=10, imgsz=640, batch=8, device=0, save_period=5)
