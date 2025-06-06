from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(data='data/dataset/data.yaml', epochs=100, imgsz=1024, batch=2, device=0)
