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
    """ Applies a region mask to isolate only the current lane. """
    h, w = image.shape[:2]  # Get height and width

    # Define a trapezoidal mask that focuses on the vehicle's lane
    lane_roi = np.array([
        (w // 2 - 300, h - 50),   # Bottom-left (closer to vehicle)
        (w // 2 - 75, h - 250),  # Upper-left (farther ahead)
        (w // 2 + 50, h - 250),  # Upper-right (farther ahead)
        (w - 50, h - 50)  # Bottom-right (closer to vehicle)
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lane_roi], 255)  # Fill the lane ROI
    
    # Convert mask to 3 channels (for RGB image)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask_rgb)
    return masked_image, mask


def preprocess_image(image, img_size=(256, 256)):
    """Prepares the image for TensorFlow Lite model inference."""
    if image is None or image.size == 0:
        return None

    #masked_image, _ = region(image)  # Apply the focused lane mask
    cropped_image = image[:, image.shape[1] // 4 : 3 * image.shape[1] // 4]  # Crop side lanes

    img = cv2.resize(cropped_image, img_size)  # Resize for model
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0).astype(np.float32)  # Model input shape


def predict_lane_single(image, interpreter):
    if image is None:
        return None

    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_index)

    if output_tensor is None or output_tensor.size == 0:
        return None

    # Convert model output to a binary mask
    pred_mask = (np.squeeze(output_tensor, axis=(0, -1)) > 0.4).astype(np.uint8) * 255

    # 🔹 Correct the mask size issue
    h, w = 256, 256  # Model output shape
    lane_mask = np.zeros((h, w), dtype=np.uint8)
    lane_mask[:, w // 4 : 3 * w // 4] = 255  # Only keep center 50% width
    pred_mask = cv2.bitwise_and(pred_mask, lane_mask)

    return pred_mask



def overlay_mask(image, mask, alpha=0.6):
    """Ensures correct overlaying of AI mask by applying it only within ROI."""
    if image is None or mask is None or mask.size == 0:
        return image

    h, w = image.shape[:2]
    
    # 🔹 Resize prediction mask to match the cropped region's size
    mask_resized = cv2.resize(mask, (w // 2, h), interpolation=cv2.INTER_NEAREST)
    
    # 🔹 Place the resized mask at the correct position in the original frame
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[:, w // 4 : 3 * w // 4] = mask_resized  # Fit within the processed ROI

    # Apply a color map for better visibility
    mask_colored = cv2.applyColorMap(full_mask, cv2.COLORMAP_HOT)
    
    # Increase brightness and contrast
    bright_mask = cv2.addWeighted(mask_colored, 1.2, np.full_like(mask_colored, 255), 0.3, 0)
    
    # Overlay mask on the original frame with better visibility
    return cv2.addWeighted(image, 1 - alpha, bright_mask, alpha, 0)



def video_reader(input_video_path):
    """Reads video frames in a separate thread and stores every 10th frame in a queue."""
    cap = cv2.VideoCapture(input_video_path)
    frame_count = 0  # Track frame index

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        if frame_count % 10 != 0:  # Skip frames that are not multiples of 10
            continue  

        if frame is None:  # Extra safety check
            continue

        frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_LINEAR)  # 🔽 Resize to 480p for faster processing
        
        if frame_queue.full():
            time.sleep(0.000000001)  # Reduce waiting time
        
        frame_queue.put(frame)  # ✅ Ensures frame is assigned before use

    cap.release()
    frame_queue.put(None)  # Signal processing thread to stop



def video_processor():
    """Processes frames from the queue and applies AI-based lane detection."""
    while True:
        frame = frame_queue.get()
        if frame is None:
            output_queue.put(None)  # Signal display thread to stop
            continue

        start_time = time.time()

        img = preprocess_image(frame)
        if img is None:
            continue
        pred_mask = predict_lane_single(img, interpreter)
        if pred_mask is None:
            continue
        overlayed_image = overlay_mask(frame, pred_mask)

        output_queue.put(overlayed_image)
        end_time = time.time()
        print(f"Processing Time per Frame: {end_time - start_time:.3f} seconds")

def video_display():
    """Displays processed frames from the output queue."""
    cv2.namedWindow("AI Processed Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Processed Video", 1920, 1080)
    #cv2.namedWindow("AI Processed Video2", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("AI Processed Video2", 640, 360)

    while True:
        frame = output_queue.get()
        if frame is None:
            continue
        image, _ = region(frame)
        cv2.imshow("AI Processed Video", frame)
        #cv2.imshow("AI Processed Video2", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_signal.set()  # Stop all threads when 'q' is pressed
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
