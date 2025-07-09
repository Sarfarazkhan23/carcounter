import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Video input
cap = cv2.VideoCapture("cars.mp4")

# Load YOLOv8 model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Use built-in class names from YOLO (more robust)
classNames = model.names

# Load region mask and overlay graphics
mask = cv2.imread("mask.png")
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)

# SORT Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Line coordinates for counting
limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if video ends or fails

    # Apply mask to focus detection region
    success, img = cap.read()
    if not success:
        break

    # Resize mask to match frame size
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Apply mask to the image
    imgRegion = cv2.bitwise_and(img, mask_resized)

    # Overlay graphics on main image
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # Run detection
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = round(float(box.conf[0]), 2)

            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Only track certain vehicle classes with confidence threshold
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker with current detections
    resultsTracker = tracker.update(detections)

    # Draw counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Draw tracking box and ID
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count logic: when object crosses the line
        if limits[0] < cx < limits[2] and (limits[1] - 15) < cy < (limits[1] + 15):
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display count on image
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Show output
    cv2.imshow("Image", img)
    # cv2.imshow("Masked Region", imgRegion)  # Optional to debug ROI
    cv2.waitKey(1)