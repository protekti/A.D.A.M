import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
import threading
import time
from queue import Queue
from PIL import Image, ImageTk

# Load the TensorFlow Lite model
tflite_model_path = "model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Shared resources to avoid frame processing issues in multiple threads
frame_lock = threading.Lock()
ai_enabled = True

# Define a buffer to store frames for batch processing
frame_buffer = []
batch_size = 4  # Process in batches of 4 frames (adjustable)

# Queue to store frames to display in Tkinter window
frame_queue = Queue()

def region(image):
    """ Applies a region mask to isolate lanes. """
    h, w = image.shape[:2]  # Get height and width
    triangle = np.array([[(150, h-50), (w//2 - 100, h//1.9), (w//2+150, h//1.9), (w-100, h-50)]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [triangle], 255)

    # Convert mask to 3 channels (for RGB image)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask_rgb)
    return masked_image, mask

def preprocess_image(image, img_size=(128, 128)):  # Reduce image size for faster inference
    img = cv2.resize(image, img_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img

def predict_lanes_batch(batch_images, interpreter):
    """ Runs inference on a batch of images. """
    batch_input = np.array(batch_images, dtype=np.float32)
    
    # Set input tensor for batch processing
    input_tensor = interpreter.tensor(input_details[0]['index'])
    input_tensor()[0:len(batch_input)] = batch_input

    # Invoke the interpreter (run inference)
    interpreter.invoke()

    # Get output tensor for the batch
    output_tensor = interpreter.tensor(output_details[0]['index'])()
    return output_tensor  # Batch of predictions

def average_lines(lines, shape):
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if slope < 0:  # Left lane
                left_lines.append((x1, y1, x2, y2))
            else:  # Right lane
                right_lines.append((x1, y1, x2, y2))
    
    def make_line(points):
        if len(points) == 0:
            return None
        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in points:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        poly = np.polyfit(x_coords, y_coords, 1)
        y1, y2 = shape[0], int(shape[0] * 0.6)
        x1, x2 = int((y1 - poly[1]) / poly[0]), int((y2 - poly[1]) / poly[0])
        return x1, y1, x2, y2
    
    left_lane = make_line(left_lines)
    right_lane = make_line(right_lines)
    
    return left_lane, right_lane

def draw_lane_lines(image, mask, shape):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lane_image = np.copy(image)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50, maxLineGap=200)
    if lines is not None:
        left_lane, right_lane = average_lines(lines, shape)
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return lane_image

def overlay_mask(image, mask):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    lane_image = draw_lane_lines(image, mask_resized, image.shape)
    return overlay, lane_image

def process_video_live(input_video_path, interpreter, ai_enabled=True):
    cap = cv2.VideoCapture(input_video_path)
    global frame_buffer
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if ai_enabled:
            region_frame, region_mask = region(frame)
            img = preprocess_image(region_frame)
            with frame_lock:
                frame_buffer.append(img)

            # Process batch when buffer reaches batch size
            if len(frame_buffer) >= batch_size:
                batch_images = frame_buffer[:batch_size]
                frame_buffer = frame_buffer[batch_size:]
                predictions = predict_lanes_batch(batch_images, interpreter)

                for idx, pred_mask in enumerate(predictions):
                    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Threshold
                    overlayed_image, lane_image = overlay_mask(frame, pred_mask)

                    # Put frame in the queue for Tkinter to display
                    frame_queue.put(overlayed_image)
                    frame_queue.put(lane_image)
        else:
            frame_queue.put(frame)
            frame_queue.put(frame)

        # Sleep a little to simulate a frame rate
        time.sleep(1 / 30)  # assuming 30 FPS for video

    cap.release()
    print("Video processing complete.")

class AI_ToggleApp:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Video Stream")
        
        self.video_label = tk.Label(master)
        self.video_label.pack()
        
        self.update_frame()

    def update_frame(self):
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Tkinter
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.video_label.config(image=frame)
            self.video_label.image = frame
        
        # Call this function after 10ms to continuously update the frame
        self.master.after(10, self.update_frame)

    def close(self):
        # Graceful shutdown
        self.master.quit()


# Example usage
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = AI_ToggleApp(root)
        
        executor = threading.Thread(target=process_video_live, args=("test2.mp4", interpreter, ai_enabled=True))
        executor.daemon = True
        executor.start()
        
        root.protocol("WM_DELETE_WINDOW", app.close)  # Properly close the Tkinter window
        root.mainloop()

    except KeyboardInterrupt:
        print("Program interrupted by user")
        cv2.destroyAllWindows()
        exit()
