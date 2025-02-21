import os
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from queue import Queue
import sys
import math

# Load TensorFlow Lite model with XNNPACK Delegate for optimized performance
tflite_model_path = "models/adam_v0.3a_350e_lite.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensor indices
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

### MULTI-THREADING IMPLEMENTATION ###
frame_queue = Queue(maxsize=10)  # Stores frames for processing
output_queue = Queue(maxsize=10)  # Stores processed frames for display
stop_signal = threading.Event()  # Used to signal threads to stop

def region(image):
    """Applies a region mask to isolate only the current lane."""
    h, w = image.shape[:2]
    lane_roi = np.array([
        (w // 2 - 1300, h - 550),
        (w // 2-400, h-900),
        (w // 2+300, h-900),
        (w - 400, h - 550)
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lane_roi], 255)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.bitwise_and(image, mask_rgb)
    return masked_image, mask

def find_lane_center(mask, original_shape):
    """Finds the left and right lane positions at the bottom and computes the center."""
    h_mask, w_mask = mask.shape[:2]
    h_orig, w_orig = original_shape[:2]
    nonzero_points = cv2.findNonZero(mask)
    if nonzero_points is None:
        return None
    midpoint = w_mask // 2
    left_lane = [pt[0][0] for pt in nonzero_points if pt[0][0] < midpoint]
    right_lane = [pt[0][0] for pt in nonzero_points if pt[0][0] > midpoint]
    if not left_lane or not right_lane:
        return None
    left_x1 = int(np.mean(left_lane))
    right_x1 = int(np.mean(right_lane))
    center_x1 = (left_x1 + right_x1) // 2
    center_y1 = h_mask - 10
    scale_x = w_orig / w_mask
    scale_y = h_orig / h_mask
    mapped_x = int(center_x1 * scale_x)
    mapped_y = int(center_y1 * scale_y)
    return mapped_x, mapped_y

def preprocess_image(image, img_size=(256, 256)):
    """Prepares the image for TensorFlow Lite model inference."""
    if image is None or image.size == 0:
        return None
    #image2, _ = region(image)
    img = cv2.resize(image, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def predict_lane_single(image, interpreter):
    """Runs the model on the given image and returns the predicted lane mask."""
    if image is None:
        return None
    #image, _ = region(image)
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_index)
    if output_tensor is None or output_tensor.size == 0:
        return None
    pred_mask = (np.squeeze(output_tensor, axis=(0, -1)) > 0.5).astype(np.uint8) * 255
    return pred_mask

def overlay_mask(image, mask, alpha=0.6):
    """Overlays both the AI raw mask and the lane area mask onto the original image."""
    if image is None or mask is None or mask.size == 0:
        return image

    # Ensure mask is resized to match the image dimensions
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Draw lane center if detected
    lane_center_X, lane_center_Y = find_lane_center(mask_resized, image.shape)

    return lane_center_X, lane_center_Y, image

def video_reader(input_video_path):
    """Reads video frames and stores every 10th frame in a queue."""
    cap = cv2.VideoCapture(input_video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:
            if frame_queue.full():
                time.sleep(.0000000000000000001)
            frame_queue.put(frame)
        frame_count += 1
    cap.release()
    frame_queue.put(None)

def video_processor():
    """Processes frames from the queue and applies AI-based lane detection."""
    while True:
        frame = frame_queue.get()
        if frame is None:
            output_queue.put(None)
            break
        img = preprocess_image(frame)
        if img is None:
            continue
        pred_mask = predict_lane_single(img, interpreter)
        if pred_mask is None:
            continue
        overlayed_image, overlayed2, image = overlay_mask(frame, pred_mask)
        output_queue.put((overlayed_image, overlayed2, image))

def drawLines(image):
    cdstP = np.copy(image)
    blur = cv2.GaussianBlur(cdstP,(15,15),0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    linesP = cv2.HoughLinesP(gray, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        # Initialize lists to store left and right lane lines
        left_lines = []
        right_lines = []
        
        # Separate lines into left and right lanes based on the slope
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / float(x2 - x1) if (x2 - x1) != 0 else 0
            
            # Classify lines as left or right based on the slope and x-position
            if slope < 0:  # Negative slope -> Left lane
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0:  # Positive slope -> Right lane
                right_lines.append((x1, y1, x2, y2))

        # Average the left and right lines if available
        def average_lines(lines):
            if len(lines) == 0:
                return None
            # Average x and y coordinates of the lines
            avg_x1 = np.mean([line[0] for line in lines])
            avg_y1 = np.mean([line[1] for line in lines])
            avg_x2 = np.mean([line[2] for line in lines])
            avg_y2 = np.mean([line[3] for line in lines])
            return (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2))

        # Calculate the average left and right lane lines
        left_avg = average_lines(left_lines)
        right_avg = average_lines(right_lines)

        # Draw the averaged left and right lane lines
        if left_avg is not None:
            cv2.line(cdstP, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (0, 0, 255), 50, cv2.LINE_AA)
        if right_avg is not None:
            cv2.line(cdstP, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (0, 0, 255), 50, cv2.LINE_AA)
    return cdstP

def video_display():
    """Displays processed frames from the output queue."""
    cv2.namedWindow("AI Processed Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Processed Video", 640, 320)
    #cv2.namedWindow("AI Processed Video2", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("AI Processed Video2", 640, 320)
    while True:
        #start = time.time()
        positionTuple = output_queue.get()
        if positionTuple is None:
            break
        laneX, laneY, frame = positionTuple
        centerX = frame.shape[0]-250
        print(laneX-centerX)
        if laneX-centerX < -30:
            print("Left")
        elif laneX-centerX > 30:
            print("Right")
        else:
            print("Center")

        #image, _ = region(frame)
        #image = drawLines(image)
        cv2.imshow("AI Processed Video", frame)
        #cv2.imshow("AI Processed Video2", image)
        #end = time.time()
        #fps = (10 + 1) / (end - start)
        #print("FPS: {:.1f}".format(fps))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit()
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        reader_thread = threading.Thread(target=video_reader, args=("test2.mp4",))
        processor_thread = threading.Thread(target=video_processor)
        display_thread = threading.Thread(target=video_display)
        reader_thread.start()
        processor_thread.start()
        display_thread.start()
        reader_thread.join()
        processor_thread.join()
        display_thread.join()
    except KeyboardInterrupt:
        print("Program interrupted by user")
        cv2.destroyAllWindows()
        sys.exit()
