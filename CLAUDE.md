# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tennis court analysis system with two approaches:

1. **Classical Computer Vision** (Existing): Hough Transform-based line detection with stabilization
2. **Deep Learning** (In Development): Keypoint R-CNN for direct court corner detection, player tracking, ball detection, and automated in/out determination

The project identifies court boundaries, players, and ball positions from video footage for automated tennis game analysis and scoring.

## Development Environment

**Conda Environment**: Use the `tennis` environment

```bash
# Activate environment
conda activate tennis

# Classical CV dependencies (already installed)
opencv-python, numpy, matplotlib, sympy

# Deep Learning dependencies (already installed)
torch, torchvision, detectron2, labelme, pycocotools

# Launch Jupyter
jupyter notebook
```

The project has no traditional build, test, or lint commands. Work is done interactively in notebooks and Python scripts.

## Architecture

### Core Processing Pipeline

1. **Frame Preprocessing** → Grayscale conversion → Gaussian blur → Binary thresholding or Canny edge detection
2. **Line Detection** → Hough Line Transform to detect line segments
3. **Line Classification** → Separate horizontal (baselines, service lines) from vertical (sidelines) based on angle
4. **Line Stabilization** → Buffer-based averaging across multiple frames to reduce jitter
5. **Perspective Transformation** → Homography to warp detected court into standardized top-down view

### Key Classes and Their Roles

**LineStabilizer2** (`court_tracker.ipynb`)
- Maintains separate deques for horizontal and vertical lines with configurable buffer size
- Merges similar lines based on proximity thresholds (10 pixels for y-axis, 5 pixels for coordinates)
- Returns stabilized lines by averaging positions across buffered frames
- Used for processing the original video perspective

**TennisCourt** (`warped_frame.ipynb`)
- Processes court from already-warped (top-down) perspective
- Uses exponential moving average (alpha parameter) for line smoothing instead of buffer
- Detects intersections between horizontal and vertical lines
- Expects nearly perfect H/V lines (threshold: <5 pixels deviation)

**Court** (`court_tracker.ipynb`)
- Simple data structure storing detected line positions
- Fields: top_baseline, bottom_baseline, top_service_line, bottom_service_line, left/right singles/doubles lines
- Also stores service_line_angle

**CourtReference** (`model-training.ipynb`)
- Defines standard tennis court dimensions based on ITF specifications
- 12 different court configurations (court_conf dictionary) for homography matching
- Dimensions: 1117px width, 2408px height for court, with 274px L/R and 549px T/B borders
- Used in homography calculation to match detected points to reference points

**Calibration** (`model-training.ipynb`)
- Interactive tool for manual ROI selection via mouse drawing
- Detects horizontal lines within the selected region
- Merges detected lines to identify service lines
- Useful for initial court boundary calibration

### Notebook Responsibilities

**court_tracker.ipynb** - Main line detection implementation
- Focus: Real-time line detection and stabilization from original video perspective
- Uses LineStabilizer2 with buffer_size=15
- Processes horizontal and vertical lines separately
- Line classification based on angle (-15° to +15° for horizontal, 60-100° for vertical)

**model-training.ipynb** - Calibration and homography tools
- Interactive corner point selection for perspective transform setup
- ROI-based calibration with Calibration class
- CourtReference model with 12 court configurations
- Homography matrix calculation using court configuration matching
- find_homography() function scores transformations and finds best match

**warped_frame.ipynb** - Post-warp processing
- Processes already-transformed top-down court view
- TennisCourt class for detecting lines from warped perspective
- Exponential moving average stabilization (alpha=0.3 default)
- Intersection detection for court keypoints

## Key Algorithms and Techniques

### Line Detection Parameters
Standard Hough Transform parameters used throughout:
- `threshold=80`: Minimum votes for line detection
- `minLineLength=50-90`: Minimum line segment length
- `maxLineGap=15-20`: Maximum gap between line segments to treat as single line

### Line Stabilization Strategy
Two approaches implemented:

1. **Buffer-based (LineStabilizer2)**: Collects lines from last N frames, merges similar ones, averages positions
2. **Exponential Moving Average (TennisCourt)**: Weighted average with alpha parameter controlling responsiveness

### Homography Matching
- Finds intersection points from all combinations of 2 horizontal + 2 vertical lines
- Compares against 12 reference court configurations
- Scores each transformation using `get_confi_score()`: correct_pixels - 0.5 * wrong_pixels
- Returns transformation matrix with highest confidence score

### Perspective Transform
Standard source points for video_input2.mp4:
```python
src_points = np.float32([[288.0, 152.0], [668.0, 150.0], [182.0, 429.0], [783.0, 428.0]])
dst_points = np.float32([[0, 0], [400, 0], [0, 500], [400, 500]])  # width=400, height=500
```

## Ground Truth and Evaluation

`ground2_truth.json` contains annotated court keypoints:
- Corner positions: top_left, top_right, bottom_left, bottom_right
- Service line corners: service_top_left, etc.

Evaluation uses pixel error calculation:
```python
error = np.linalg.norm(np.array(detected_point) - np.array(ground_truth_point))
```

## Important Implementation Details

### Frame Processing Standard
Most processing uses 960x540 resolution:
```python
frame = cv2.resize(frame, (960, 540))
```

### Binary Thresholding for Court Lines
Threshold value of 200 isolates white court lines:
```python
gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
```

### Line Merging Logic
Horizontal lines merge if:
- Y-coordinate difference < 10 pixels
- X-ranges overlap (with 20 pixel tolerance)

Vertical lines merge if all coordinates differ by < 5 pixels.

## Deep Learning Pipeline (In Development)

### Goal
Build complete tennis analysis system: court detection → player tracking → ball tracking → bounce detection → in/out calls → scoring

### Phase 1: Court Keypoint Detection (Current)
**Approach**: Train Keypoint R-CNN to directly predict 8 court keypoints
- More robust than Hough Transform (handles occlusions, shadows, worn lines)
- End-to-end learning instead of hand-crafted rules

**Workflow**:
1. Extract frames: `python extract_frames.py` (extracts diverse frames from videos)
2. Annotate keypoints: `labelme training_data/frames` (mark 8 keypoints per frame)
3. Convert to COCO: `python convert_to_coco.py` (creates COCO keypoint dataset)
4. Train model: TBD (Detectron2 training script)
5. Evaluate: Compare with Hough Transform baseline

**Keypoints** (8 total, annotate in this order):
1. top_left_baseline
2. top_right_baseline
3. bottom_left_baseline
4. bottom_right_baseline
5. top_left_service
6. top_right_service
7. bottom_left_service
8. bottom_right_service

See `ANNOTATION_GUIDE.md` for detailed annotation instructions.

### Future Phases
- **Phase 2**: Player detection using YOLO (yolo11n.pt already available)
- **Phase 3**: Ball detection (TrackNet or custom CNN)
- **Phase 4**: Ball tracking and bounce detection (physics-based + Kalman filter)
- **Phase 5**: In/out determination (geometric calculation with homography)
- **Phase 6**: Game logic and scoring (tennis rules state machine)

### Key Concepts to Understand
- **Keypoint R-CNN**: Extension of Mask R-CNN for keypoint detection
- **Transfer Learning**: Start with COCO pre-trained weights, fine-tune on tennis courts
- **COCO Format**: Standard annotation format for object detection/keypoints
- **Homography**: Perspective transformation from camera view to top-down court coordinates
- **Tracking**: Connect detections across frames (DeepSORT/ByteTrack)

## Files Not to Modify

- `yolo11n.pt`: Pre-trained YOLO model weights (5.6MB) for player detection
- `ground2_truth.json`: Ground truth annotations for classical CV evaluation
- `VideoInput/`: Sample video files for testing
- `training_data/frames/`: Extracted frames for annotation (generated)
- `training_data/tennis_court_keypoints.json`: COCO dataset (generated)