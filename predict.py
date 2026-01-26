from ultralytics import YOLO
import cv2

# 1. Load your best trained weights
model = YOLO('models/best.pt') 

# 2. Run prediction on a test image
# Replace 'test.jpg' with an actual path to an MRI image
results = model.predict(source='test.jpg', imgsz=224, conf=0.5)

# 3. Display the top prediction
for r in results:
    print(f"Prediction: {r.names[r.probs.top1]} ({r.probs.top1conf.item()*100:.2f}%)")
