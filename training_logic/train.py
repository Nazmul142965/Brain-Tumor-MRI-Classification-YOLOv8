from ultralytics import YOLO

# Load the lightweight YOLOv8-Nano model
model = YOLO('yolov8n-cls.pt') 

# Start training using the data.yaml configuration
results = model.train(
    data='data.yaml', 
    epochs=20, 
    imgsz=224,
    project='brain_tumor_mri',
    name='v8_nano_experiment'
)
