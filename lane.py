import cv2
import numpy as np
from numpy.polynomial import Polynomial as P

# Suppress warnings for polyfit if needed
np.seterr(all='ignore')

def grey(image):
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gauss(image):
    return cv2.GaussianBlur(image, (7,7), 0)

def canny(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

def region(image):
    h, w = image.shape
    triangle = np.array([ 
        [(0, h), (900, 500), (w, h)] 
    ])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

# Perspective Transform: Define source and destination points
def perspective_transform(image):
    h, w = image.shape[:2]
    # Define points for perspective transform (source points)
    src_points = np.float32([(150, h-80), (250, 200), (w-180, h-80)])
    # Define the destination points for the top-down perspective
    dst_points = np.float32([(0, h), (0, 0), (w, 0), (w, h)])
    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Apply the perspective warp to the image
    warped_image = cv2.warpPerspective(image, M, (w, h))
    return warped_image, M

def average(image, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))

    # Compute the average for both left and right lines
    left_avg = np.mean(left, axis=0) if left else None
    right_avg = np.mean(right, axis=0) if right else None
    
    return left_avg, right_avg

def make_points(image, average): 
    if average is None:
        return None  # Return None if no valid line found
    slope, y_int = average 
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    
    # Ensure no division by zero
    if slope != 0:
        x1 = int((y1 - y_int) / slope) 
        x2 = int((y2 - y_int) / slope)
    else:
        # If slope is zero, we simply use a vertical line at the center
        x1 = int(image.shape[1] / 2)
        x2 = x1

    return np.array([x1, y1, x2, y2])

def display_lines(image, left_line, right_line):
    try:
        lines_image = np.zeros_like(image)
        if left_line is not None:
            cv2.line(lines_image, (int(left_line[0]), int(left_line[1])), (int(left_line[0]), int(left_line[2])), (255, 0, 0), 10)
        if right_line is not None:
            cv2.line(lines_image, (int(right_line[0]), int(right_line[1])), (int(right_line[0]), int(right_line[2])), (0, 0, 255), 10)
        return lines_image
    except:
        return None

# Video processing code
video_input = "test.mp4"
cap = cv2.VideoCapture(video_input)

# Check if the video is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video details
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize VideoWriter to save output video
out = cv2.VideoWriter('output_video_perspective.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply perspective transform to get a top-down view
    #warped_frame, M = perspective_transform(frame)

    # Apply lane detection methods on the warped frame
    copy = np.copy(frame)
    grey_img = grey(copy)
    gaus = gauss(grey_img)
    edges = canny(gaus)
    isolated = region(edges)

    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(isolated, rho=2, theta=np.pi/180, threshold=35, minLineLength=20, maxLineGap=5)

    if lines is not None:
        averaged_lines = average(copy, lines)

        # Display the lines on the warped frame
        black_lines = display_lines(copy, *averaged_lines)

        # Show the processed frame in a window
        cv2.namedWindow("Lanes", cv2.WINDOW_NORMAL) 
        cv2.imshow('Lanes', copy)
        cv2.resizeWindow("Lanes", 640, 360) 
    else:
        print("No lines detected.")

    cv2.namedWindow("Lanes4", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("Lanes2", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("Lanes3", cv2.WINDOW_NORMAL) 

    cv2.resizeWindow("Lanes4", 640, 360) 
    cv2.resizeWindow("Lanes2", 640, 360) 
    cv2.resizeWindow("Lanes3", 640, 360) 
    cv2.imshow('Lanes4', isolated)
    cv2.imshow('Lanes2', edges)
    cv2.imshow('Lanes3', frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
