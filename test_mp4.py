import cv2
import numpy as np
import torch
from dev import ENet

def region(image):
    """ Applies a region mask to isolate lanes. """
    h, w = image.shape[:2]  # Get height and width

    # Define a polygonal region (Adjust coordinates as needed)
    triangle = np.array([ 
        [(0, h), (w//2 - 100, h//2), (w//2 + 100, h//2), (w, h)]
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


def test_video(video_path):
    """ Performs real-time lane detection on an MP4 video using OpenCV. """
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ENet(2, 4).to(device)
    model.load_state_dict(torch.load("adam_v0.1a.pth", map_location=device))
    model.eval()

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        original_frame = frame.copy()

        # Convert the frame to grayscale for region mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply the region mask to grayscale frame (just to isolate lane region)
        gray = region(gray)

        # Show the AI's view (what it sees as the region of interest)
        cv2.imshow("AI's View (Region Mask)", gray)

        # Resize the frame
        resized_frame = cv2.resize(gray, (512, 256))

        # Normalize and convert to PyTorch tensor
        # Ensure tensor is [batch_size, channels, height, width]
        input_tensor = torch.from_numpy(resized_frame).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

        # Run inference
        with torch.no_grad():
            binary_output, instance_output = model(input_tensor)

        # Apply sigmoid activation to get probability values
        binary_mask = torch.sigmoid(binary_output)

        # Set a higher threshold for stricter lane detection
        dynamic_threshold = binary_mask[0, 0].mean().item() + 0.2  # Increase margin
        dynamic_threshold = min(dynamic_threshold, 0.7)  # Cap at 0.7

        # Apply threshold and convert to NumPy array
        binary_mask = (binary_mask > 0.8).float().cpu().numpy()  # Use a fixed threshold instead

        # Convert binary mask to 0-255 for visualization
        binary_mask = (binary_mask * 255).astype(np.uint8)

        # Apply color map for visualization
        lanes_overlay = cv2.applyColorMap(binary_mask[0, 0], cv2.COLORMAP_JET)

        # Resize overlay to match original frame
        lanes_overlay_resized = cv2.resize(lanes_overlay, (original_frame.shape[1], original_frame.shape[0]))

        # Blend overlay with the resized 3-channel grayscale frame
        result_frame = cv2.addWeighted(original_frame, 0.7, lanes_overlay_resized, 0.3, 0)

        # Display the final result
        cv2.imshow("Lane Detection", result_frame)

        # Press 'Q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------
# MAIN FUNCTION
# -------------------------
if __name__ == "__main__":
    video_path = "test2.mp4"  # Change this to your video file
    test_video(video_path)
