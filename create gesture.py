import cv2
import numpy as np
import os
import time  # Added for delay

# Directory to save gesture data
DATA_DIR = r"C:\Users\udupa\OneDrive\Desktop\gesture\train"
os.makedirs(DATA_DIR, exist_ok=True)

# Background settings
background = None
accumulated_weight = 0.5

# Region of Interest (ROI)
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return
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

def collect_gestures(gesture_name, num_samples=300):
    cam = cv2.VideoCapture(0)
    num_frames = 0
    num_imgs_taken = 0
    delay_started = False

    # Create folder for this gesture
    gesture_path = os.path.join(DATA_DIR, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)

    print("[INFO] Starting data collection...")
    print(f"[INFO] Saving images to: {gesture_path}")
    print("[INFO] Press 'Esc' to exit early.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if num_frames < 60:
            cal_accum_avg(gray, accumulated_weight)
            cv2.putText(frame_copy, "Fetching background... Please wait", (70, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        elif num_frames == 60 and not delay_started:
            print("[INFO] Background captured.")
            print("[INFO] Starting gesture capture in 10 seconds...")
            cv2.putText(frame_copy, "Get ready! Starting in 10 seconds...", (50, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Gesture Collection", frame_copy)
            cv2.waitKey(1)
            time.sleep(10)
            delay_started = True

        else:
            hand = segment_hand(gray)
            if hand is not None:
                thresholded, hand_segment = hand
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

                if num_imgs_taken < num_samples:
                    img_path = os.path.join(gesture_path, f"{num_imgs_taken}.jpg")
                    cv2.imwrite(img_path, thresholded)
                    num_imgs_taken += 1

                cv2.imshow("Thresholded", thresholded)
                cv2.putText(frame_copy, f"{gesture_name}: {num_imgs_taken}/{num_samples}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw ROI
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (0, 255, 255), 2)
        cv2.imshow("Gesture Collection", frame_copy)

        num_frames += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or num_imgs_taken >= num_samples:  # ESC to quit
            break

    print("[INFO] Data collection complete.")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    gesture_name = input("Enter gesture name (e.g., 'one', 'two', etc.): ").strip().lower()
    collect_gestures(gesture_name)
