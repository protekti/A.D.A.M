import carla
import numpy as np
import cv2
import tensorflow as tf
import time
import threading
from queue import Queue
import sys
import random

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Spawn a vehicle
blueprint = random.choice(world.get_blueprint_library().filter('model3'))
spawn_point = carla.Transform(carla.Location(x=160, y=-375, z=0.281942), carla.Rotation(yaw=0))
world.get_spectator().set_transform(spawn_point)
print(f"Spawning at: {spawn_point.location}")
vehicle = world.spawn_actor(blueprint, spawn_point)

# Create a camera sensor attached to the vehicle
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1920')
camera_bp.set_attribute('image_size_y', '1080')
camera_transform = carla.Transform(carla.Location(x=2.5, z=2))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="adam_v0.3a_350e_lite.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

frame_queue = Queue(maxsize=10)
output_queue = Queue(maxsize=10)

# PID Constants
Kp = 0.00000000001  # Proportional gain
Ki = 0.00000000001  # Integral gain
Kd = 1  # Derivative gain

prev_error = 0
integral = 0

def control_vehicle(vehicle, throttle=0, steer=0.0, brake=0.0):
    control = carla.VehicleControl()
    control.throttle = throttle
    control.steer = steer
    control.brake = brake
    vehicle.apply_control(control)

def compute_steering(lane_center_x, image_width):
    """Computes steering based on PID control to reduce oscillations."""
    global prev_error, integral

    image_center = image_width / 2
    error = lane_center_x - image_center  # Deviation from center

    # PID calculations
    integral += error  # Accumulate error
    derivative = error - prev_error  # Compute change in error
    prev_error = error  # Store error for next iteration

    # Compute steering value using PID formula
    steer = (Kp * error) + (Ki * integral) + (Kd * derivative)
    steer *= 0.5

    # Clip the steering value to the allowed range [-0.5, 0.5]
    steer = np.clip(steer, -0.1, 0.1)

    print(f"Lane Center X: {lane_center_x}, Image Center: {image_center}")
    print(f"Error: {error}, Integral: {integral}, Derivative: {derivative}")
    print(f"PID Steering: {steer}")

    return steer

def region(image):
    """Applies a region mask to isolate only the current lane."""
    h, w = image.shape[:2]
    lane_roi = np.array([  # Define a region of interest (ROI) for the lane
        (w-100, h),
        (w // 2 + 100, h - 350),
        (w // 2 - 150, h - 350),
        (100, h)
    ], dtype=np.int32)
    
    # Create a blank mask with the same dimensions as the image
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Fill the mask with the polygon (lane region) in white (255)
    cv2.fillPoly(mask, [lane_roi], 255)
    
    # Convert the mask to 3-channel (RGB) format
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Resize the mask to match the image dimensions
    mask_rgb_resized = cv2.resize(mask_rgb, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Debug: Check sizes before applying bitwise_and
    print(f"Image shape: {image.shape}, Mask shape: {mask_rgb_resized.shape}")
    
    # Perform bitwise AND operation to isolate the lane region in the image
    if image.shape == mask_rgb_resized.shape:
        masked_image = cv2.bitwise_and(image, mask_rgb_resized)  # Apply the mask
    else:
        print(f"Shape mismatch: image {image.shape} and mask {mask_rgb_resized.shape}")
        masked_image = image  # Fallback to original image if there's a shape mismatch
    
    return masked_image, mask


def find_lane_center(mask, original_shape):
    """Finds the left and right lane positions at the bottom and computes the center."""
    h_mask, w_mask = mask.shape[:2]
    h_orig, w_orig = original_shape[:2]
    nonzero_points = cv2.findNonZero(mask)
    if nonzero_points is None:
        return 0, 0
    midpoint = w_mask // 2
    left_lane = [pt[0][0] for pt in nonzero_points if pt[0][0] < midpoint]
    right_lane = [pt[0][0] for pt in nonzero_points if pt[0][0] > midpoint]
    if not left_lane or not right_lane:
        return 0, 0
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
    image, _ = region(image)
    img = cv2.resize(image, img_size) / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def predict_lane_single(image, interpreter):
    """Runs the model on the given image and returns the predicted lane mask."""
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_index)
    if output_tensor is None or output_tensor.size == 0:
        return None
    pred_mask = (np.squeeze(output_tensor, axis=(0, -1)) > 0.3).astype(np.uint8) * 255
    return pred_mask



def overlay_mask(image, mask, alpha=0.6):
    """Overlays both the AI raw mask and the lane area mask onto the original image."""
    if image is None or mask is None or mask.size == 0:
        return image
    # Resize mask to match the image dimensions
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Find the lane center (this function might also need to handle size mismatch)
    lane_center_X, lane_center_Y = find_lane_center(mask_resized, image.shape)
    
    # Overlay the mask on the image
    processed_img = cv2.addWeighted(image, 1, cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR), alpha, 0)
    
    return lane_center_X, lane_center_Y, processed_img


def process_frame():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        img = preprocess_image(frame)
        pred_mask = predict_lane_single(img, interpreter)
        lane_center_x, _, processed_img = overlay_mask(frame, pred_mask)

        # Compute steering based on lane center
        steer_value = compute_steering(lane_center_x, frame.shape[1])
        control_vehicle(vehicle, throttle=0.2, steer=steer_value, brake=0.0)

        output_queue.put(processed_img)

def display_frames():
    cv2.namedWindow("AI Processed Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Processed Video", 640, 320)
    cv2.namedWindow("Crop", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crop", 640, 320)
    while True:
        processed_frame = output_queue.get()
        if processed_frame is None:
            break
        cv2.imshow("AI Processed Video", processed_frame)
        cropped, _ = region(processed_frame)
        cv2.imshow("Crop", cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def on_image_received(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    image_rgb = array[:, :, :3]
    frame_queue.put(image_rgb)

processing_thread = threading.Thread(target=process_frame)
display_thread = threading.Thread(target=display_frames)
processing_thread.start()
display_thread.start()

camera.listen(on_image_received)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    camera.stop()
    vehicle.destroy()
    frame_queue.put(None)
    output_queue.put(None)
    processing_thread.join()
    display_thread.join()
    cv2.destroyAllWindows()
