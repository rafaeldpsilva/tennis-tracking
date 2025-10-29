# Tennis Court Tracking

A computer vision project for detecting and tracking tennis court lines from video footage using OpenCV and Python. This system automatically identifies court boundaries, service lines, and key intersection points to create a stabilized top-down view of the tennis court.

## Features

- **Automated Court Line Detection**: Uses Hough Line Transform to detect horizontal and vertical court lines from video frames
- **Line Stabilization**: Smooths detected lines across multiple frames to reduce jitter and false detections
- **Perspective Transformation**: Applies homography to warp the detected court into a standardized top-down view
- **Interactive Calibration**: Manual ROI selection tool for initial court boundary calibration
- **Court Reference Model**: Built-in tennis court model with standard dimensions and line positions
- **Intersection Detection**: Automatically calculates intersection points of detected court lines
- **Real-time Processing**: Processes video frames in real-time with visual feedback

## Project Structure

```
tennis-tracking/
├── court_tracker.ipynb          # Main court line detection and stabilization
├── model-training.ipynb         # Training utilities and calibration tools
├── warped_frame.ipynb          # Perspective transformation and top-down view
├── ground2_truth.json          # Ground truth annotations for evaluation
├── yolo11n.pt                  # YOLO model weights for object detection
└── VideoInput/                 # Sample tennis video files
    ├── video_input2.mp4
    ├── video_input3.mp4
    └── ...
```

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib
- SymPy
- Collections

Install dependencies:
```bash
pip install opencv-python numpy matplotlib sympy
```

## How It Works

### 1. Line Detection
The system uses a multi-step pipeline to detect court lines:
- Converts video frames to grayscale
- Applies Gaussian blur to reduce noise
- Uses Canny edge detection to identify edges
- Applies Hough Line Transform to detect line segments
- Classifies lines as horizontal or vertical based on angle

### 2. Line Stabilization
Two stabilization approaches are implemented:

**Buffer-based Stabilization** (`LineStabilizer2`):
- Maintains a buffer of detected lines from recent frames
- Merges similar lines based on proximity
- Averages line positions across the buffer for stable output

**Exponential Moving Average**:
- Smooths line positions using weighted averaging
- Reduces sudden jumps while maintaining responsiveness

### 3. Perspective Transformation
The system can transform the detected court into a top-down view:
- Uses manually selected corner points or automatically detected intersections
- Computes homography matrix to warp the perspective
- Creates a standardized court view for analysis

### 4. Court Reference Model
Includes a `CourtReference` class with standard tennis court dimensions:
- Baseline positions (top and bottom)
- Service line positions
- Singles and doubles sidelines
- Net position and center service line
- 12 different court configurations for homography matching

## Usage

### Basic Court Detection

```python
import cv2
import numpy as np

# Load video
video_path = "VideoInput/video_input2.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize line stabilizer
line_stabilizer = LineStabilizer2(buffer_size=15)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect lines
    horizontal, vertical = detect_lines(blurred)

    # Draw detected lines
    for line in horizontal:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow("Court Line Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Interactive Calibration

```python
from calibration import Calibration

# Load first frame
cap = cv2.VideoCapture("VideoInput/video_input2.mp4")
ret, first_frame = cap.read()
first_frame = cv2.resize(first_frame, (960, 540))

# Run calibration tool
calibrator = Calibration(first_frame)
calibrator.run()

# Get detected service line
service_line = calibrator.bottom_service_line
```

### Perspective Transformation

```python
# Define source points (court corners in original view)
src_points = np.float32([
    [288.0, 152.0],   # Top-left
    [668.0, 150.0],   # Top-right
    [182.0, 429.0],   # Bottom-left
    [783.0, 428.0]    # Bottom-right
])

# Define destination points (top-down view)
width, height = 400, 500
dst_points = np.float32([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
])

# Compute transformation matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply transformation
warped_frame = cv2.warpPerspective(frame, M, (width, height))
```

## Key Components

### LineStabilizer2
Buffers detected lines across frames and merges similar lines to produce stable output.

**Parameters**:
- `buffer_size` (int): Number of frames to buffer (default: 5)

**Methods**:
- `add_lines(h_lines, v_lines)`: Add detected lines from current frame
- `get_h_stable_lines()`: Get stabilized horizontal lines
- `get_v_stable_lines()`: Get stabilized vertical lines

### Court
Stores detected court line positions and metadata.

**Attributes**:
- `service_line_angle`: Angle of service line
- `lines`: Dictionary containing positions of all court lines

### CourtReference
Provides standard tennis court dimensions and reference points for homography matching.

**Methods**:
- `build_court_reference()`: Generate reference court image
- `get_important_lines()`: Return all court line positions
- `get_court_mask()`: Get masked regions of the court

## Evaluation

The system includes ground truth annotations for accuracy evaluation:

```json
{
    "1": {
        "top_left": [288.0, 151.0],
        "top_right": [667.0, 151.0],
        "bottom_left": [182.0, 428.0],
        "bottom_right": [782.0, 428.0],
        "service_top_left": [324.0, 196.0],
        ...
    }
}
```

Calculate pixel error between detected and ground truth points to measure accuracy.

## Notebooks

### court_tracker.ipynb
Main implementation of court line detection with advanced stabilization techniques. Focuses on detecting and tracking horizontal and vertical lines in real-time.

### model-training.ipynb
Contains various tools and utilities:
- Interactive corner point selection
- Simple line detector focusing on central region
- ROI-based calibration
- Homography transformation experiments

### warped_frame.ipynb
Implements the `TennisCourt` class for processing warped court views and detecting lines from a top-down perspective.

## Limitations

- Requires good video quality with clear court lines
- Works best with static camera angles
- May struggle with shadows, worn lines, or complex backgrounds
- Initial calibration may be needed for different camera angles

## Future Improvements

- Automatic court detection without manual calibration
- Player and ball tracking integration
- Real-time court boundary prediction
- Support for multiple camera angles
- Machine learning-based line detection

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- OpenCV community for computer vision tools
- Tennis court reference dimensions based on ITF standards
