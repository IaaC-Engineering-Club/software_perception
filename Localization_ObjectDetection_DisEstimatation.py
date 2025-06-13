import os
import time
import cv2
import tkinter as tk
from tkinter import filedialog
from pwm_controller import PWMController
from qgc_plan_converter import convert_csv_to_plan
from mission_logger import MissionLogger
import numpy as np

# === CONFIG ===
CAMERA_INDEX = 0
MAVLINK_UDP = "udp:127.0.0.1:14551"
PLAN_OUTPUT = "mission.plan"
LOG_FILE_PATH = "mission_log.txt"

# YOLO config files - make sure to have these in your working dir
YOLO_CONFIG = "yolov4-tiny.cfg"
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_CLASSES = "coco.names"

# Real object height in meters (e.g., average buoy height)
REAL_OBJECT_HEIGHT = 0.3  # meters, adjust to your objects

# Approximate focal length in pixels (camera specific, you may need to calibrate)
FOCAL_LENGTH = 700  # pixels, tweak for accuracy

logger = MissionLogger()

def ask_for_csv_path():
    root = tk.Tk()
    root.withdraw()
    print(" Select GPS CSV file or press Cancel to skip.")
    file_path = filedialog.askopenfilename(
        title="Select GPS CSV File (Optional)",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_path:
        print("⚠️ CSV skipped. No mission.plan will be created.")
        return None
    print(f"✅ Selected file: {file_path}")
    return file_path

class visionNav:
    def __init__(self, video):
        self.cap = video
        self.image = None
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Load YOLO network
        self.net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
        self.layer_names = self.net.getUnconnectedOutLayersNames()
        self.classes = open(YOLO_CLASSES).read().strip().split("\n")
        
        self.detections = []

    def detect_objects_with_distance(self):
        blob = cv2.dnn.blobFromImage(self.image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.layer_names)

        h, w = self.image.shape[:2]
        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter for confident detections
                if confidence > 0.5:
                    box = detection[0:4] * np.array([w, h, w, h])
                    center_x, center_y, width, height = box.astype("int")
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-max suppression to reduce overlapping boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        self.detections = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w_box, h_box = boxes[i]
                label = self.classes[class_ids[i]]

                # Estimate distance using pinhole camera model
                distance = (REAL_OBJECT_HEIGHT * FOCAL_LENGTH) / h_box if h_box > 0 else -1

                self.detections.append((x, y, w_box, h_box, label, distance))

                # Draw bounding box and distance text
                cv2.rectangle(self.image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                text = f"{label}: {distance:.2f} m"
                cv2.putText(self.image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def interpret_decision(nav: visionNav):
    # Use detections to decide how to steer
    if not nav.detections:
        return "TURN_AROUND"

    # Take the detection closest to the center of the frame
    middle_of_frame = nav.width // 2
    closest_detection = None
    closest_offset = nav.width  # large initial value

    for (x, y, w, h, label, dist) in nav.detections:
        mid_x = x + w // 2
        offset = abs(mid_x - middle_of_frame)
        if offset < closest_offset:
            closest_offset = offset
            closest_detection = (x, y, w, h, label, dist)

    if closest_detection is None:
        return "TURN_AROUND"

    mid_x = closest_detection[0] + closest_detection[2] // 2

    if abs(mid_x - middle_of_frame) < nav.width * 0.1:
        return "KEEP_ROUTE"
    elif mid_x < middle_of_frame:
        return "TURN_LEFT"
    else:
        return "TURN_RIGHT"

def handle_decision(decision, pwm: PWMController):
    logger.log(f" Decision: {decision}")

    if decision == "KEEP_ROUTE":
        pwm.go_forward()
    elif decision == "TURN_LEFT":
        pwm.steer_left()
    elif decision == "TURN_RIGHT":
        pwm.steer_right()
    elif decision == "TURN_AROUND":
        pwm.steer_left()
        time.sleep(1.5)
        pwm.steer_right()
        time.sleep(1.5)
    else:
        pwm.stop_all()

def main():
    print(" NJORD Autonomous Boat Control - UDP Mode")
    csv_path = ask_for_csv_path()
    if csv_path:
        convert_csv_to_plan(csv_path, PLAN_OUTPUT)
        logger.log(f"✅ CSV converted to {PLAN_OUTPUT}")
    else:
        logger.log("⚠️ No CSV path provided. Continuing without mission plan.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        exit(1)

    nav = visionNav(video=cap)
    pwm = PWMController(MAVLINK_UDP)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            nav.image = frame
            nav.detect_objects_with_distance()

            decision = interpret_decision(nav)
            handle_decision(decision, pwm)

            cv2.imshow("VisionNav Debug", nav.image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n🛑 Mission interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pwm.stop_all()
        logger.log(" Mission ended")
        logger.save_to_file(LOG_FILE_PATH)

if __name__ == "__main__":
    main()
