# Player Detection with Mask R-CNN

Complete guide for tennis player detection using Mask R-CNN instance segmentation.

## Overview

The player detection system uses Detectron2's pre-trained Mask R-CNN model to:
- ✅ Detect players in video frames
- ✅ Generate pixel-level segmentation masks
- ✅ Track player positions on court
- ✅ Filter detections to players on court only

## Files

### `detect_players.py`
Standalone player detection using Mask R-CNN.

**Features:**
- `PlayerDetector` class for easy integration
- Configurable confidence thresholds
- Multiple model architectures (R50-FPN, R101-FPN, X101-FPN)
- Visualization with bounding boxes and masks
- Court-based filtering support

**Usage:**

```bash
# Test on single image
python detect_players.py --image test_frame.jpg --output result.jpg --confidence 0.7

# Process full video
python detect_players.py --video VideoInput/video.mp4 --output output/players.mp4

# Use more accurate model (slower)
python detect_players.py --video input.mp4 --model R101-FPN --confidence 0.8
```

### `court_player_analysis.py`
Integrated court + player detection and analysis.

**Features:**
- Combined visualization of court lines and players
- Court-based player filtering (removes spectators)
- Player positioning analysis
- Foundation for ball tracking and in/out calls

**Usage:**

```bash
# Run demo with manual court coordinates
python court_player_analysis.py --demo

# Process video with custom settings
python court_player_analysis.py --video input.mp4 --output result.mp4 --confidence 0.6

# Enable court filtering (requires court keypoints)
python court_player_analysis.py --video input.mp4 --filter-court
```

## Player Detection API

### Basic Detection

```python
from detect_players import PlayerDetector

# Initialize detector
detector = PlayerDetector(confidence_threshold=0.7)

# Detect players in frame
frame = cv2.imread('frame.jpg')
detections = detector.detect(frame)

print(f"Found {detections['num_players']} players")
print(f"Boxes: {detections['boxes']}")      # Bounding boxes [x1, y1, x2, y2]
print(f"Masks: {detections['masks']}")      # Binary masks (H x W)
print(f"Scores: {detections['scores']}")    # Confidence scores
```

### Court-Based Filtering

```python
# Create court mask (1s inside court, 0s outside)
court_mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(court_mask, [court_polygon_points], 1)

# Detect only players on court
detections = detector.detect_and_filter_by_court(
    frame,
    court_mask=court_mask,
    min_overlap=0.3  # At least 30% of player must be on court
)
```

### Player Positioning

```python
# Get player center points
centers = detector.get_player_centers(detections)
# Returns: [(x1, y1), (x2, y2), ...]

# Get player foot positions (useful for court positioning)
feet = detector.get_player_foot_positions(detections)
# Returns: [(x1, y2_bottom), (x2, y2_bottom), ...]
```

### Visualization

```python
# Visualize detections
annotated = detector.visualize(
    frame,
    detections,
    show_masks=True,    # Draw segmentation masks
    show_boxes=True,    # Draw bounding boxes
    show_labels=True    # Draw labels with confidence
)

cv2.imshow('Players', annotated)
```

## Integrated Analysis

```python
from court_player_analysis import TennisAnalyzer

# Initialize analyzer
analyzer = TennisAnalyzer(player_confidence=0.7)

# Set court keypoints (from court detection)
court_keypoints = {
    'top_left_baseline': (100, 200),
    'top_right_baseline': (900, 200),
    # ... more keypoints
}
analyzer.set_court_keypoints(court_keypoints)

# Analyze single frame
annotated, analysis = analyzer.analyze_frame(
    frame,
    draw_court=True,
    draw_players=True,
    filter_by_court=True  # Only show players on court
)

print(f"Players on court: {analysis['num_players']}")
print(f"Positions: {analysis['player_positions']}")

# Process full video
analyzer.process_video(
    video_path="input.mp4",
    output_path="output.mp4",
    filter_by_court=True
)
```

## Model Architecture Options

### R50-FPN (Default) ⚡
- **Speed**: ~10-15 FPS on CPU, 30+ FPS on GPU
- **Accuracy**: Good
- **Model size**: 178 MB
- **Use case**: Real-time processing, general use

### R101-FPN 🎯
- **Speed**: ~8-12 FPS on CPU, 25+ FPS on GPU
- **Accuracy**: Better
- **Model size**: 193 MB
- **Use case**: When accuracy is more important than speed

### X101-FPN 🔬
- **Speed**: ~5-8 FPS on CPU, 20+ FPS on GPU
- **Accuracy**: Best
- **Model size**: 240 MB
- **Use case**: Offline analysis, maximum accuracy needed

## Performance Tips

### Speed Optimization
1. **Use GPU**: Set `CUDA_VISIBLE_DEVICES=0` if GPU available
2. **Reduce resolution**: Resize frames before detection
3. **Increase confidence**: Higher threshold = fewer detections to process
4. **Use R50-FPN**: Fastest model while maintaining good accuracy

### Accuracy Optimization
1. **Lower confidence**: Capture more marginal detections
2. **Use X101-FPN**: Most accurate model
3. **Post-process**: Use temporal smoothing across frames
4. **Fine-tune**: Train on tennis-specific dataset for better results

## Integration with Court Detection

The player detection system is designed to integrate with the court keypoint detection:

```python
# Future integration example
from detect_court import CourtDetector  # To be implemented
from detect_players import PlayerDetector
from court_player_analysis import TennisAnalyzer

# Initialize
court_detector = CourtDetector()
analyzer = TennisAnalyzer()

# Process frame
frame = cv2.imread('frame.jpg')

# Detect court
court_keypoints = court_detector.detect(frame)
analyzer.set_court_keypoints(court_keypoints)

# Detect players on court
annotated, analysis = analyzer.analyze_frame(frame, filter_by_court=True)
```

## Next Steps

### Phase 2: Player Tracking ✨
Add temporal tracking to assign consistent IDs across frames:
- **DeepSORT**: Deep learning + Kalman filter tracking
- **ByteTrack**: State-of-the-art multi-object tracking
- **Benefits**: Track player movements, analyze rally patterns

### Phase 3: Ball Detection 🎾
- Use TrackNet or custom CNN for ball detection
- Integrate with player positions
- Detect ball-player interactions

### Phase 4: Game Logic 📊
- Bounce detection (ball hits court)
- In/out determination (using court geometry)
- Scoring system (tennis rules state machine)
- Rally analysis

## Troubleshooting

### Issue: Too many false positives (spectators detected)
**Solution**: Enable court filtering
```python
detections = detector.detect_and_filter_by_court(frame, court_mask, min_overlap=0.5)
```

### Issue: Missing players
**Solution**: Lower confidence threshold
```python
detector = PlayerDetector(confidence_threshold=0.5)  # Default: 0.7
```

### Issue: Slow performance
**Solution**: Use faster model or reduce resolution
```python
# Resize frame
frame = cv2.resize(frame, (960, 540))
# Use faster model
detector = PlayerDetector(model_type="R50-FPN")
```

### Issue: Model download fails
**Solution**: Download manually
```bash
# Download from Detectron2 model zoo
wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
# Place in ~/.torch/fvcore_cache/detectron2/
```

## Technical Details

### Detection Pipeline
1. **Input**: BGR image (numpy array)
2. **Preprocessing**: Image normalization, GPU transfer
3. **Backbone**: ResNet feature extraction
4. **RPN**: Region proposal network generates candidate boxes
5. **ROI Heads**: Refines boxes, classifies, generates masks
6. **NMS**: Non-maximum suppression removes duplicates
7. **Output**: Boxes, masks, scores for 'person' class

### Mask Format
- **Shape**: (H, W) boolean array per detection
- **Values**: True = player pixel, False = background
- **Usage**: Can be used for precise positioning, occlusion analysis

### Coordinate Format
- **Bounding boxes**: [x1, y1, x2, y2] (top-left, bottom-right)
- **Keypoints**: (x, y) pixel coordinates
- **Origin**: Top-left corner of image

## References

- [Detectron2 Documentation](https://detectron2.readthedocs.io/)
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [COCO Dataset](https://cocodataset.org/)
