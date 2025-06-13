import cv2
import torch
import time
import json
import os

# === CONFIGURATION ===
CAMERA_INDEX = 0  # Camera input index
FOCAL_LENGTH_PIXELS = 960  # Focal length (you may need to calibrate this)

# Approximate real-world height (in meters) for each class
REAL_OBJECT_HEIGHT = {
    0: 0.2,   # Person (face to shoulder)
    8: 0.15   # Small boat (example)
}

# === Initialize YOLOv5 Model ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.classes = [0, 8]  # Only detect person (0) and boat (8)
COCO_CLASSES = model.names.copy()
COCO_CLASSES[0] = "person"
COCO_CLASSES[8] = "boat"

# === Utility Functions ===
def estimate_distance(real_height, focal_length, pixel_height):
    if pixel_height == 0:
        return None
    return (real_height * focal_length) / pixel_height

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    os.makedirs("detections", exist_ok=True)
    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            timestamp = time.time()

            results = model(frame)
            detections = results.xyxy[0]  # [x1, y1, x2, y2, confidence, class]

            detection_list = []
            has_detection = False

            for *box, conf, cls in detections:
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls.item())
                height_in_pixels = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                real_height = REAL_OBJECT_HEIGHT.get(class_id)
                distance = estimate_distance(real_height, FOCAL_LENGTH_PIXELS, height_in_pixels) if real_height else None

                # Estimated 3D coordinates (x, y, z) — assuming image center as origin
                x_relative = center_x - frame.shape[1] // 2
                y_relative = center_y - frame.shape[0] // 2
                z_distance = round(distance, 2) if distance else None

                label_text = f"{COCO_CLASSES[class_id]} ({conf:.2f})"
                if z_distance:
                    label_text += f" {z_distance:.2f}m"
                label_text += f" [{center_x},{center_y}]"

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save detection
                detection_list.append({
                    "class": COCO_CLASSES[class_id],
                    "confidence": float(conf),
                    "center": {"x": center_x, "y": center_y},
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "position_estimate": {
                        "x": x_relative,
                        "y": y_relative,
                        "z": z_distance
                    }
                })

                has_detection = True

            # Save only frames with detections
            if has_detection:
                output_data = {
                    "frame": frame_counter,
                    "timestamp": timestamp,
                    "detections": detection_list
                }
                json_path = f"detections/frame_{frame_counter:05d}.json"
                with open(json_path, "w") as f_json:
                    json.dump(output_data, f_json, indent=2)

            # Show image
            cv2.imshow("YOLO Maritime Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
