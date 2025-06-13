import cv2
import torch
import time

# === CONFIG ===
CAMERA_INDEX = 0
REAL_OBJECT_HEIGHT = 0.6  # in meters (e.g., buoy height)
FOCAL_LENGTH_PIXELS = 960  # this depends on your camera and resolution

# Load YOLOv5 model (install with: pip install torch torchvision yolov5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#model.classes = [0]  # For example, only detect 'person' or set to class ID of buoy

def estimate_distance(real_height, focal_length, pixel_height):
    if pixel_height == 0:
        return None
    return (real_height * focal_length) / pixel_height

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detections = results.xyxy[0]  # [x1, y1, x2, y2, confidence, class]

            for *box, conf, cls in detections:
                x1, y1, x2, y2 = map(int, box)
                height_in_pixels = y2 - y1
                distance = estimate_distance(REAL_OBJECT_HEIGHT, FOCAL_LENGTH_PIXELS, height_in_pixels)

                label = f"Buoy: {distance:.2f}m" if distance else "Unknown"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLO Distance Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
