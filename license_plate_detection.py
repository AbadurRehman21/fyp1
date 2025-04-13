import os
import cv2
import time
from collections import Counter
from ultralytics import YOLO
import easyocr
import re  # Import regex module

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define path for the license plate detection model
trained_model_path = "license.pt"

# Load YOLO model for license plate detection
try:
    model = YOLO(trained_model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define province names to filter out
provinces = {"PUNJAB", "SINDH", "KPK", "BALOCHISTAN", "ISLAMABAD"}

# Predefined license plate formats (regex patterns)
# Single-line formats: "ABC-123" or "ABC 123"
# Double-line formats: "ABC-12 1234", "ABC 1234", or "ABC 12 1234"
license_plate_formats = [
    r"^[A-Z]{3}[- ]?\d{3}$",  # Single-line format (ABC-123 or ABC 123)
    r"^[A-Z]{3}[- ]?\d{2} \d{4}$",  # Double-line format (ABC-12 1234)
    r"^[A-Z]{3} \d{4}$",  # Double-line format (ABC 1234)
    r"^[A-Z]{3} \d{2} \d{4}$",  # Double-line format (ABC 12 1234)
]

# Global counter to store detected license plates
license_plate_counts = Counter()

def refine_license_text(text):
    """
    Refines the detected license plate text by:
    1. Removing province names.
    2. Cleaning unwanted characters.
    3. Matching against predefined formats.
    
    Args:
        text (str): The raw OCR-detected text.
        
    Returns:
        str: The refined license plate text.
    """
    # Remove province names
    for province in provinces:
        text = text.replace(province, "")
    
    # Remove unwanted characters and spaces
    cleaned_text = re.sub(r"[^A-Z0-9-]", "", text.upper())
    
    # Try to match the cleaned text with predefined formats
    for pattern in license_plate_formats:
        match = re.match(pattern, cleaned_text)
        if match:
            return match.group(0)  # Return the matched text
    
    # If no format matches, return the cleaned text as-is
    return cleaned_text

def process_license_plate_frame(frame):
    """
    Processes a single frame for license plate detection.
    This function:
      - Runs the YOLO model to detect license plate regions.
      - Extracts those regions and uses EasyOCR to perform OCR.
      - Draws bounding boxes and recognized text onto the frame.
      - Updates a global counter of detected license plates.
      
    Args:
        frame (numpy.ndarray): The input image/frame.
        
    Returns:
        numpy.ndarray: The processed frame with drawn detections.
    """
    results = model.predict(frame, conf=0.5)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            license_plate = frame[y1:y2, x1:x2]

            try:
                ocr_results = reader.readtext(license_plate)
                extracted_texts = [res[1].strip().upper() for res in ocr_results]  # Convert to uppercase
                
                # Join extracted texts and refine using predefined formats
                raw_text = " ".join(extracted_texts)
                license_text = refine_license_text(raw_text)
                
                if license_text:
                    license_plate_counts[license_text] += 1
            except Exception as e:
                print(f"OCR Error: {e}")
                license_text = "OCR Error"

            # Draw bounding box and text on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, license_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Detected License Plate: {license_text}")
    
    return frame

# Optional: If you run this module directly, you can add a simple test.
if __name__ == "__main__":
    # For testing purposes only; adjust test_input_path as needed.
    test_input_path = r"path_to_a_test_video_or_image"
    if test_input_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")):
        frame = cv2.imread(test_input_path)
        if frame is None:
            print(f"Error: Could not open image {test_input_path}")
            exit()
        processed_frame = process_license_plate_frame(frame)
        cv2.imshow("License Plate Detection", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(test_input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {test_input_path}")
            exit()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_license_plate_frame(frame)
            cv2.imshow("License Plate Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
