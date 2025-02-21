import os
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from queue import Queue

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
        (w // 2 - 1100, h - 375),
        (w // 2-400, h-1100),
        (w // 2+150, h-1100),
        (w - 400, h - 375)
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
    return (mapped_x, mapped_y)

def preprocess_image(image, img_size=(256, 256)):
    """Prepares the image for TensorFlow Lite model inference."""
    if image is None or image.size == 0:
        return None
    image, _ = region(image)
    img = cv2.resize(image, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def predict_lane_single(image, interpreter):
    """Runs the model on the given image and returns the predicted lane mask."""
    if image is None:
        return None
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_index)
    if output_tensor is None or output_tensor.size == 0:
        return None
    pred_mask = (np.squeeze(output_tensor, axis=(0, -1)) > 0.4).astype(np.uint8) * 255
    return pred_mask

def fill_lane_area(image, mask):
    """Fills the area between the detected lane lines, keeping the rest of the image black."""
    nonzero_points = cv2.findNonZero(mask)
    if nonzero_points is None:
        return np.zeros_like(image)  # Return a completely black image if no lane is detected

    hull = cv2.convexHull(nonzero_points)
    filled_mask = np.zeros_like(mask)
    cv2.fillPoly(filled_mask, [hull], 255)

    # Resize the filled mask to match the original image dimensions
    filled_mask = cv2.resize(filled_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create an output image with only the filled lane area in green, rest black
    output = np.zeros_like(image)
    output[filled_mask == 255] = (0, 255, 0)  # Green for lane area

    return output



def overlay_mask(image, mask, alpha=0.6):
    """Overlays the raw AI mask onto the original image and draws lane center."""
    if image is None or mask is None or mask.size == 0:
        return image
    lane_area = fill_lane_area(image, mask)
    overlayed = cv2.addWeighted(image, 1, lane_area, 0.5, 0)
    mask_resized = cv2.resize(overlayed, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlayed = cv2.addWeighted(image, 1 - alpha, mask_resized, alpha, 0)
    lane_center = find_lane_center(mask, image.shape)
    if lane_center:
        cv2.circle(overlayed, lane_center, 10, (0, 255, 0), -1)
    return overlayed

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
                time.sleep(0.001)
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
        overlayed_image = overlay_mask(frame, pred_mask)
        output_queue.put(overlayed_image)

def video_display():
    """Displays processed frames from the output queue."""
    cv2.namedWindow("AI Processed Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Processed Video", 640, 320)
    while True:
        start = time.time()
        frame = output_queue.get()
        if frame is None:
            break
        cv2.imshow("AI Processed Video", frame)
        end = time.time()
        fps = (10 + 1) / (end - start)
        print("FPS: {:.1f}".format(fps))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_signal.set()
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        reader_thread = threading.Thread(target=video_reader, args=("long test.mp4",))
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
        exit()
