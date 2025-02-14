import numpy as np
import cv2

def region(image):
    h, w, temp = image.shape
    triangle = np.array([ 
        [(0, h-100), (860, 500),  (1060, 500), (w, h-100)] 
    ])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def HSV_color_selection(image):
    """
    Apply color selection to the HSV images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    imageCopy = image.copy()
    imageCopy = cv2.GaussianBlur(image, (7, 7), 0) 
    imageCopy = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #White color mask
    lower_white = np.array([0,0,168])
    upper_white = np.array([172,111,255])
    white_mask = cv2.inRange(imageCopy, lower_white, upper_white)
    
    #Yellow color mask
    lower_threshold = np.array([80, 80, 18])
    upper_threshold = np.array([160, 160, 36])
    yellow_mask = cv2.inRange(imageCopy, lower_threshold, upper_threshold)
    
    #Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(imageCopy, imageCopy, mask = mask)

    regionImage = region(masked_image)
    
    return regionImage

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

def canny(image):
    edges = cv2.Canny(image, 100, 150)
    return edges

def display_lines(image, line):
    cv2.line(image, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (255, 0, 0), 10)
    return image

cap = cv2.VideoCapture('test.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()

    copy = frame.copy()

    colorSelect = HSV_color_selection(copy)
    edges = canny(colorSelect)
    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=2, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
    x1 = lines[0][0][0]
    x2 = lines[0][0][2]
    y1 = lines[0][0][1]
    y2 = lines[0][0][3]
    print(x1, x2, y1, y2)
    if lines is not None:
        averaged_lines = average(copy, lines)

        # Display the lines on the warped frame
        black_lines = display_lines(copy, [x1, y1, x2, y2])

        cv2.namedWindow("Lanes", cv2.WINDOW_NORMAL) 
        cv2.imshow('Lanes', black_lines)
        cv2.resizeWindow("Lanes", 640, 360) 
    else:
        print("No lines detected.")

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("frame", 640, 360)
    cv2.namedWindow("frame2", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("frame2", 640, 360)
    cv2.namedWindow("frame3", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("frame3", 640, 360)

    cv2.imshow('frame',frame)
    cv2.imshow('frame2',colorSelect)
    cv2.imshow('frame3',edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
