import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Load video
cap = cv2.VideoCapture("video.mp4")

# Initialize car counts for left and right directions
car_count_left = 0
car_count_right = 0

car_counted = False  # Flag to track if a car has been counted after passing the line
right_crossed = False
left_crossed = False

# Define the x-coordinate of the vertical line
line_x = 400




# Initialize flags and variables
car_crossing = False
car_direction = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Extract vehicle detections
    detections = results.pred[0]
    for detection in detections:
        label = int(detection[5])  # Class label for vehicle
        confidence = detection[4]

        if label == 2 and confidence > 0.6:  # Class label 2 corresponds to "car"
            x1, y1, x2, y2 = detection[:4].int().tolist()  # Convert to list of integers
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Car {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            car_center_x = (x1 + x2) / 2

            if x1 < line_x < x2:  # Car center is crossing the line
                if not car_crossing:
                    car_crossing = True
                    if car_center_x > line_x:
                        car_direction = "left"
                    else:
                        car_direction = "right"
            else:
                if car_crossing:
                    car_crossing = False
                    if car_direction == "left":
                        car_count_left += 1
                        print("Car crossed from right to left")
                    elif car_direction == "right":
                        car_count_right += 1
                        print("Car crossed from left to right")
                    car_direction = None

    

    

    # Display car counts for left and right directions
    cv2.putText(frame, f"Left Count: {car_count_left}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Right Count: {car_count_right}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the vertical line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)

    cv2.imshow("Vehicle Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Break the loop when 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
