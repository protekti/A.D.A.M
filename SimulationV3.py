import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("latest_best_model.keras")

def region(image):
    """ Applies a region mask to isolate lanes. """
    h, w = image.shape[:2]  # Get height and width

    # Define a polygonal region (Adjust coordinates as needed)
    triangle = np.array([ 
        [(0, h), (w//2 - 70, h//1.7), (w//2 + 70, h//1.7), (w, h)]
    ], dtype=np.int32)

    # Create a blank mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Fill the polygon (ROI) with white (255)
    cv2.fillPoly(mask, [triangle], 255)

    # Check if the image is grayscale (1 channel) or color (3 channels)
    if len(image.shape) == 2:  # Grayscale
        # Convert the grayscale image to 3 channels
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:  # Color image (3 channels)
        image_rgb = image

    # Convert mask to 3 channels (for RGB image)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image_rgb, mask_rgb)

    return masked_image

# Function to preprocess a single image
def preprocess_image(image, img_size=(256, 256)):
    img = cv2.resize(image, img_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img

# Function to make predictions
def predict_lanes(image, model):
    image = region(image)  # Apply region mask
    img = preprocess_image(image)
    pred_mask = model.predict(img)[0]  # Get the first (and only) prediction
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Threshold the mask
    return pred_mask

# Function to extract lane lines from the predicted mask and draw them
def draw_lane_lines(image, mask):
    # Find contours in the predicted mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw the lane lines
    lane_image = np.copy(image)
    
    # Iterate over the contours and draw lines
    for contour in contours:
        if cv2.contourArea(contour) > 300:  # Ignore small contours
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw the polygon (lane lines)
            cv2.polylines(lane_image, [approx], isClosed=False, color=(0, 255, 0), thickness=5)
    
    return lane_image

# Function to overlay the predicted mask on the original image and draw lane lines
def overlay_mask(image, mask):
    # Resize mask to match the original image size
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Apply color map to mask (convert grayscale to BGR)
    mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    
    # Overlay the mask on the original image
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    
    # Draw lane lines on the overlay
    lane_image = draw_lane_lines(image, mask_resized)
    
    return overlay, lane_image

# Function to process a video and display the output live
def process_video_live(input_video_path, model):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make prediction on the current frame
        predicted_mask = predict_lanes(frame, model)
        
        # Overlay the predicted mask and draw lane lines
        overlayed_image, lane_image = overlay_mask(frame, predicted_mask)
        
        # Display the frame with overlay and lane lines
        cv2.imshow('Video with Lane Detection (Overlay)', overlayed_image)
        cv2.imshow('Video with Lane Detection (Lane Lines)', lane_image)
        
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")

# Example usage
input_video_path = "test2.mp4"  # Change this to your input video path

process_video_live(input_video_path, model)
