# vehicle_detection.py
import os
from ultralytics import YOLO
import cv2

# Define the path to the vehicle detection model and confidence threshold
VEHICLE_MODEL_PATH = "yolov8s.pt"
VEHICLE_CONF_THRESH = 0.5

# Load the YOLOv8 model pre-trained on the COCO dataset
try:
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
except Exception as e:
    print(f"Error loading vehicle detection model: {e}")
    raise e

# Set of COCO class names that correspond to vehicles
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}

def detect_vehicle(frame):
    """
    Runs vehicle detection on the input frame using the YOLOv8 model.
    
    Parameters:
        frame (numpy.ndarray): The input image/frame.
    
    Returns:
        list of dict: A list where each dictionary contains:
            - 'box': [x1, y1, x2, y2] coordinates of the bounding box,
            - 'confidence': Confidence score for the detection,
            - 'class': The label "vehicle" (regardless of the actual vehicle type).
    """
    results = vehicle_model.predict(frame, conf=VEHICLE_CONF_THRESH)
    detections = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_indices = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls_idx in zip(boxes, confs, cls_indices):
            # Get the original class name from the model
            original_class = vehicle_model.names[int(cls_idx)]
            # Check if the detection belongs to one of our vehicle classes
            if original_class in VEHICLE_CLASSES:
                detections.append({
                    "box": list(map(int, box)),
                    "confidence": float(conf),
                    "class": "vehicle"  # Override the label to "vehicle"
                })
    return detections

# Optional: Test the module directly.
if __name__ == "__main__":
    test_image_path = "path_to_test_vehicle_image.jpg"  # Update with a valid image path
    frame = cv2.imread(test_image_path)
    if frame is None:
        print("Error: Could not load test image.")
    else:
        dets = detect_vehicle(frame)
        for det in dets:
            print(det)
            # Draw bounding box and unified "vehicle" label
            box = det["box"]
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Vehicle Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
