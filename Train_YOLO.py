from ultralytics import YOLO


model = YOLO("yolov8n-seg.pt")
results = model.train(data="/home/moo/PycharmProjects/Train_Yolo/train.yaml", epochs=150, imgsz=640, batch=64)
