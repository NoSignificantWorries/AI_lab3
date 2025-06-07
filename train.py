from ultralytics import YOLO

# model = YOLO('yolov8s.pt')
model = YOLO('runs/detect/train3/weights/best.pt')

# model.train(data='data/dataset/data.yaml', epochs=100, imgsz=640, batch=8, device=0, save_period=5)
model.train(data='data/many_dataset/data.yaml', epochs=50, imgsz=640, batch=8, device=0, save_period=5)
