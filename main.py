import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model (e.g., a YOLO model for object detection)
model = tf.keras.models.load_model('"F:\Mining\BlastCaptain\Data1of3\C1_352_108\352_108.MP4"')

# Load the video
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_frame = cv2.resize(frame, (416, 416))  # Resize to model input size
    input_frame = input_frame / 255.0  # Normalize
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension

    # Perform object detection
    detections = model.predict(input_frame)

    # Post-process detections (this will depend on your model's output format)
    # For example, if using YOLO, you might need to filter out low-confidence detections
    # and apply non-max suppression

    # Visualize detections on the frame
    for detection in detections:
        # Extract bounding box coordinates and confidence
        x, y, w, h, confidence = detection
        if confidence > 0.5:  # Threshold for confidence
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()