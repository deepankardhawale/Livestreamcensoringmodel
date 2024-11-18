

import cv2
import numpy as np
import threading
import queue
from collections import deque

# Load YOLO model and configuration
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set up VideoCapture
cap = cv2.VideoCapture(input("Enter file name or press enter to start webcam: ") or 0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

frame_queue = queue.Queue(maxsize=10)  # Queue for storing frames
detection_history = deque(maxlen=10)  # History for temporal stability

# Frame control variables
frame_skip = 5  # Number of frames to skip in idle mode
frame_counter = 0  # Counter to track frames
is_weapon_detected = False  # Flag for weapon detection state


def capture_frames():
    """Capture frames from video source and store in queue."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get()  # Remove oldest frame if queue is full
        frame_queue.put(frame)


# Start frame capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

while True:
    if frame_queue.empty():
        continue  # Wait until a frame is available in the queue

    img = frame_queue.get()  # Get the next frame from the queue
    frame_counter += 1

    # Skip frames dynamically based on detection state
    if frame_counter % frame_skip != 0:
        continue

    # Resize frame for faster processing
    img = cv2.resize(img, (640, 360))
    height, width, _ = img.shape

    # Detecting objects with YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Parse YOLO detections
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Class probabilities
            class_id = np.argmax(scores)  # Get class with max probability
            confidence = scores[class_id]  # Get confidence level
            if confidence > 0.5:  # Filter out low-confidence detections
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x, y = max(0, center_x - w // 2), max(0, center_y - h // 2)
                w, h = min(width - x, w), min(height - y, h)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    is_weapon_detected = len(indexes) > 0  # Check if a weapon is detected

    # Store detections in history for temporal stability
    if is_weapon_detected:
        detection_history.append([(boxes[i], class_ids[i]) for i in indexes.flatten()])
    else:
        detection_history.append([])

    # Combine detections from history
    combined_boxes = []
    for frame_boxes in detection_history:
        combined_boxes.extend(frame_boxes)

    # Filter out redundant bounding boxes
    final_boxes = []
    for box, class_id in combined_boxes:
        if not any(np.allclose(box, b, atol=20) for b, _ in final_boxes):
            final_boxes.append((box, class_id))

    # Blur and annotate detected regions
    for (x, y, w, h), class_id in final_boxes:
        roi = img[y:y + h, x:x + w]
        if roi.size > 0:
            img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (15, 15), 30)  # Apply Gaussian blur
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_id], 2)  # Draw bounding box
        cv2.putText(img, str(classes[class_id]), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, colors[class_id], 3)

    # Adjust frame skip dynamically based on detection state
    frame_skip = 1 if is_weapon_detected else 5  # Process all frames if weapon detected, otherwise skip frames

    # Display processed frame
    cv2.imshow("Weapon Detection", img)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
