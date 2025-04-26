# pyright: reportMissingImports=false

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model(r"C:\Users\udupa\Downloads\gesture recognition project\gesture_recognition_model.keras")

# Dynamically get class labels from model's training
# Assume same train path: D:/gesture/train
train_path = r"C:\Users\udupa\OneDrive\Desktop\gesture\train"
class_labels = sorted(os.listdir(train_path))
class_indices = {i: label for i, label in enumerate(class_labels)}

# Initialize camera and parameters
background = None
accumulated_weight = 0.5
ROI_top, ROI_bottom, ROI_right, ROI_left = 100, 300, 150, 350

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment

def predict_gesture():
    cam = cv2.VideoCapture(0)
    num_frames = 0

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 60:
            cal_accum_avg(gray_frame, accumulated_weight)
            cv2.putText(frame_copy, "Calibrating Background...", (80, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else:
            hand = segment_hand(gray_frame)
            if hand is not None:
                thresholded, hand_segment = hand
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 2)

                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.expand_dims(thresholded, axis=0)
                thresholded = tf.keras.applications.vgg16.preprocess_input(thresholded)

                pred = model.predict(thresholded)
                gesture = class_indices[np.argmax(pred)]
                confidence = np.max(pred)

                cv2.putText(frame_copy, f"Gesture: {gesture}", (170, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame_copy, f"Confidence: {confidence:.2f}", (170, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.imshow("Thresholded", thresholded[0])

        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
        num_frames += 1
        cv2.putText(frame_copy, "Real-time Gesture Recognition", (10, 20), 
                   cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
        cv2.imshow("Gesture Recognition", frame_copy)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_gesture()
