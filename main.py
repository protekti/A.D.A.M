import cv2
import numpy as np
import threading
from ultralytics import YOLO
import sys
import os

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt", verbose=False)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# ========== Lane Detection Functions ==========
def convert_hsl(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def HSL_color_selection(image):
    converted = convert_hsl(image)
    white_mask = cv2.inRange(converted, (0, 200, 0), (255, 255, 255))
    yellow_mask = cv2.inRange(converted, (0, 0, 100), (40, 255, 255))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_smoothing(image, kernel_size=3):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_detector(image, low_threshold=25, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_selection(image):
    mask = np.zeros_like(image)
    rows, cols = image.shape[:2]
    vertices = np.array([[
        (cols * 0.1, rows * 0.8), 
        (cols * 0.4, rows * 0.55),
        (cols * 0.5, rows * 0.55), 
        (cols * 0.9, rows * 0.8)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return mask

def hough_transform(image):
    return cv2.HoughLinesP(image, 1, np.pi/180, 10, minLineLength=20, maxLineGap=300)

def average_lines(lines, shape):
    left_lines = []
    right_lines = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
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

def draw_lane_lines(image, lines, mask):
    line_image = np.zeros_like(image)
    if lines is not None:
        left_lane, right_lane = average_lines(lines, image.shape)
        if left_lane:
            cv2.line(line_image, left_lane[:2], left_lane[2:], (0, 0, 255), 5)
        if right_lane:
            cv2.line(line_image, right_lane[:2], right_lane[2:], (0, 0, 255), 5)
    masked_lines = cv2.bitwise_and(line_image, line_image, mask=mask)
    return cv2.addWeighted(image, 1, masked_lines, 1, 0)

def frame_processor(image):
    color_select = HSL_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    edges = canny_detector(smooth)
    mask = region_selection(edges)
    region = cv2.bitwise_and(edges, mask)
    hough = hough_transform(region)
    lane_lines = draw_lane_lines(image, hough, mask)
    return lane_lines

# ========== Car Detection ==========
def detect_cars(image):
    with SuppressOutput():
        results = model(image)
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = map(float, box.tolist())  # Ensure values are floats
        if int(cls) in [2, 7]:  # Car and truck classes
            box_height = y2 - y1
            distance = (1.5 * 300) / box_height  # Approximate distance calculation
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{distance:.2f}m", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return image


# ========== Multithreading Functions ==========
def process_lane_detection(frame, lane_result):
    lane_frame = frame_processor(frame)
    lane_result.append(lane_frame)

def process_car_detection(frame, car_result):
    car_frame = detect_cars(frame)
    car_result.append(car_frame)

def process_video_live(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        lane_result, car_result = [], []
        
        lane_thread = threading.Thread(target=process_lane_detection, args=(frame, lane_result))
        car_thread = threading.Thread(target=process_car_detection, args=(frame, car_result))
        
        lane_thread.start()
        car_thread.start()
        
        lane_thread.join()
        car_thread.join()
        
        if lane_result and car_result:
            combined_frame = cv2.addWeighted(lane_result[0], 0.7, car_result[0], 0.3, 0.0)
            combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Lane & Car Detection", combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_live("test2.mp4")
