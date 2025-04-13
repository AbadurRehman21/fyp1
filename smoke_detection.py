import cv2
from ultralytics import YOLO

SMOKE_MODEL_PATH = "smoke.pt"
SMOKE_CONF_THRESH = 0.3

try:
    smoke_model = YOLO(SMOKE_MODEL_PATH)
except Exception as e:
    print(f"Error loading smoke model: {e}")
    raise e

def detect_smoke(frame):
    results = smoke_model.predict(source=frame, conf=SMOKE_CONF_THRESH, save=False, verbose=False)
    smoke_detected = False
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box, cls_idx, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                if int(cls_idx) == 0 and conf >= SMOKE_CONF_THRESH:
                    smoke_detected = True
                    x1, y1, x2, y2 = map(int, box)
                    label = f"Smoke {conf:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return smoke_detected, annotated_frame
