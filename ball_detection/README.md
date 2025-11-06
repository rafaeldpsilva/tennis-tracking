# TrackNet Ball Detection 🎾

State-of-the-art tennis ball detection using TrackNet deep learning architecture.

## Overview

TrackNet is a deep learning model specifically designed for detecting small, fast-moving objects in sports videos. It uses **temporal information from 3 consecutive frames** to handle motion blur and predict ball positions with high accuracy.

### Key Features

✅ **Temporal Context**: Uses 3-frame sequences to handle motion blur
✅ **Heatmap Output**: More robust than bounding boxes for tiny objects
✅ **U-Net Architecture**: Encoder-decoder with skip connections
✅ **Two Model Sizes**: Full (44MB, accurate) vs Lightning (3.6MB, fast)
✅ **Interactive Annotation**: GUI tool for labeling ball positions
✅ **Complete Pipeline**: Training, inference, and visualization

## Architecture

**Input**: 3 consecutive RGB frames stacked → 9 channels
**Output**: Heatmap where peak = ball position

```
Encoder (VGG16-style)    Decoder (U-Net style)
   9 channels                  Skip connections
       ↓                              ↑
   [Conv + BN + ReLU] ←────────→ [Upsample + Conv]
       ↓                              ↑
   [MaxPool]              →  [Concatenate]
       ↓                              ↑
   [512 channels]         →  [256 → 128 → 64 → 1]
```

### Model Variants

| Model | Parameters | Size | Speed | Use Case |
|-------|-----------|------|-------|----------|
| **TrackNet** | 11.6M | 44MB | ~10 FPS | Best accuracy |
| **TrackNetLightning** | 951K | 3.6MB | ~30 FPS | Fast inference |

## Installation

```bash
# Already installed in tennis environment
conda activate tennis

# Required packages (already in environment)
# - torch
# - torchvision
# - opencv-python
# - pandas
# - matplotlib
# - tqdm
```

## Quick Start

### 1. Annotate Ball Positions

```bash
# Interactive annotation tool
python ball_detection/annotate_ball.py \
  --video VideoInput/video_input6.mp4 \
  --output ball_detection/ball_annotations/video6_annotations.csv \
  --sample-rate 1  # Annotate every frame (or use 30 for 1/sec)
```

**Controls:**
- Click on ball to mark position
- `v` - Mark ball as not visible
- `n` or Space - Next frame
- `b` or Backspace - Previous frame
- `s` - Save annotations
- `q` or ESC - Save and quit

### 2. Train Model

```bash
python ball_detection/train_tracknet.py \
  --video VideoInput/video_input6.mp4 \
  --annotations ball_detection/ball_annotations/video6_annotations.csv \
  --output output/tracknet \
  --epochs 50 \
  --batch-size 4 \
  --model full  # or 'light' for faster model
```

Training takes ~1-2 hours on CPU, ~15-30 minutes on GPU for 50 epochs.

### 3. Detect Ball in Videos

```bash
python ball_detection/detect_ball.py \
  --video VideoInput/test_video.mp4 \
  --model output/tracknet/best_model.pth \
  --output output/ball_detected.mp4 \
  --threshold 0.5
```

## Detailed Usage

### Annotation

#### Interactive Annotation

```bash
# Full frame-by-frame annotation
python ball_detection/annotate_ball.py \
  --video input.mp4 \
  --output annotations.csv

# Annotate every 10th frame (faster)
python ball_detection/annotate_ball.py \
  --video input.mp4 \
  --output annotations.csv \
  --sample-rate 10

# Start from specific frame
python ball_detection/annotate_ball.py \
  --video input.mp4 \
  --output annotations.csv \
  --start-frame 500
```

#### Batch Template Creation

```bash
# Create annotation templates for all videos
python ball_detection/annotate_ball.py \
  --batch VideoInput/ \
  --batch-output ball_annotations/ \
  --batch-sample 30  # 1 frame per second at 30fps
```

#### Annotation Format (CSV)

```csv
frame_id,x,y,visibility,orig_width,orig_height
0,512,384,1,1920,1080
1,515,390,1,1920,1080
2,-1,-1,0,1920,1080  # Ball not visible
```

- `frame_id`: Frame number
- `x, y`: Ball position in pixels
- `visibility`: 1 = visible, 0 = not visible
- `orig_width, orig_height`: Original video resolution

### Training

#### Basic Training

```bash
python ball_detection/train_tracknet.py \
  --video input.mp4 \
  --annotations annotations.csv \
  --epochs 50
```

#### Advanced Options

```bash
python ball_detection/train_tracknet.py \
  --video input.mp4 \
  --annotations annotations.csv \
  --output output/tracknet_custom \
  --epochs 100 \
  --batch-size 8 \  # Larger batch (needs more GPU memory)
  --lr 0.0001 \  # Learning rate
  --model light \  # Use lighter model
  --val-split 0.2  # 20% validation split
```

#### Training Output

```
output/tracknet/
├── best_model.pth           # Best model by validation loss
├── final_model.pth          # Final model after all epochs
├── checkpoint_epoch_10.pth  # Checkpoints every 10 epochs
├── checkpoint_epoch_20.pth
└── training_curves.png      # Loss and accuracy plots
```

### Inference

#### Basic Detection

```bash
python ball_detection/detect_ball.py \
  --video input.mp4 \
  --model output/tracknet/best_model.pth \
  --output result.mp4
```

#### Advanced Options

```bash
python ball_detection/detect_ball.py \
  --video input.mp4 \
  --model output/tracknet/best_model.pth \
  --output result.mp4 \
  --model-type light \  # Use light model
  --threshold 0.3 \  # Lower threshold (more detections, more false positives)
  --trajectory-length 50 \  # Longer trajectory trail
  --no-viz  # Faster (no display window)
```

## Python API

### Model Creation

```python
from ball_detection.tracknet_model import TrackNet, TrackNetLightning

# Create model
model = TrackNet()  # Full model
# or
model = TrackNetLightning()  # Light model

# Test forward pass
import torch
frames = torch.randn(1, 9, 288, 512)  # Batch of 3-frame sequences
heatmap = model(frames)  # Output: (1, 1, 288, 512)

# Get ball position
position = model.predict_ball_position(heatmap, threshold=0.5)
if position:
    x, y, confidence = position
    print(f"Ball at ({x}, {y}) with confidence {confidence:.2f}")
```

### Training

```python
from ball_detection.train_tracknet import train_tracknet

train_tracknet(
    video_path="input.mp4",
    annotation_csv="annotations.csv",
    output_dir="output/tracknet",
    epochs=50,
    batch_size=4,
    learning_rate=1e-4,
    model_type='full',
    val_split=0.2
)
```

### Inference

```python
from ball_detection.detect_ball import BallDetector

# Create detector
detector = BallDetector(
    model_path="output/tracknet/best_model.pth",
    model_type='full',
    threshold=0.5
)

# Process video
detections = detector.process_video(
    video_path="input.mp4",
    output_path="output.mp4",
    show_visualization=True,
    draw_trajectory=True
)

# Access detections
for det in detections:
    if det['x'] >= 0:
        print(f"Frame {det['frame']}: Ball at ({det['x']}, {det['y']}) conf={det['conf']:.2f}")
```

### Dataset

```python
from ball_detection.tracknet_dataset import TrackNetDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = TrackNetDataset(
    video_path="input.mp4",
    annotation_csv="annotations.csv",
    width=512,
    height=288,
    sigma=5.0  # Gaussian sigma for heatmap
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate
for frames, heatmaps in dataloader:
    # frames: (batch, 9, H, W)
    # heatmaps: (batch, 1, H, W)
    pass
```

## How It Works

### 1. Temporal Information

Traditional object detection processes single frames, but tennis balls move extremely fast and create motion blur. TrackNet uses **3 consecutive frames** to capture temporal context:

```
Frame t-1    Frame t    Frame t+1
   🎾    →      🎾    →      🎾
    \___________|___________/
          |
    3-frame sequence
    (motion context)
```

### 2. Heatmap Regression

Instead of predicting a bounding box (which is difficult for 5-10 pixel balls), TrackNet predicts a **probability heatmap**:

```
Input: 3 frames → Model → Output: Heatmap

     [Image]                 [Heatmap]
      . . .                   0 0 0 0
      . 🎾 .        →         0 1 0 0  ← Peak = ball position
      . . .                   0 0 0 0
```

### 3. Training Process

1. Load 3 consecutive frames
2. Create Gaussian heatmap at annotated ball position
3. Train model to predict heatmap
4. Loss: MSE or Focal Loss (handles class imbalance)

### 4. Inference Process

1. Load 3-frame sequence
2. Run through model → get heatmap
3. Find peak in heatmap
4. If peak > threshold → ball detected
5. Repeat for next frame

## Performance Tips

### Accuracy

- **Annotate more data**: 100+ frames with diverse ball positions
- **Annotate carefully**: Precise ball center, not approximate
- **Include hard cases**: Ball in shadows, near lines, motion blur
- **Mark not visible**: Don't force annotations when ball is off-screen
- **Use full model**: TrackNet vs TrackNetLightning
- **Train longer**: 50-100 epochs minimum
- **Lower threshold**: Capture more detections (but more false positives)

### Speed

- **Use GPU**: 3-5x faster training and inference
- **Use light model**: TrackNetLightning (91% fewer parameters)
- **Larger batch size**: 8-16 on GPU (needs more memory)
- **Skip frames**: Process every Nth frame for real-time
- **Reduce resolution**: 256x144 instead of 512x288

### Data Requirements

| Scenario | Frames to Annotate | Expected Accuracy |
|----------|-------------------|-------------------|
| **Quick test** | 50-100 | 60-70% |
| **Decent model** | 200-500 | 75-85% |
| **Production** | 1000+ | 85-95% |

## Troubleshooting

### Issue: Poor detection accuracy

**Causes**:
- Insufficient training data
- Ball annotations not precise
- Model underfitting

**Solutions**:
```bash
# Annotate more frames
python ball_detection/annotate_ball.py --video input.mp4 --output annotations.csv

# Train longer
python ball_detection/train_tracknet.py --epochs 100

# Use full model (not light)
python ball_detection/train_tracknet.py --model full

# Lower detection threshold
python ball_detection/detect_ball.py --threshold 0.3
```

### Issue: Many false positives

**Causes**:
- Threshold too low
- Model confusing lines/objects with ball

**Solutions**:
```bash
# Increase threshold
python ball_detection/detect_ball.py --threshold 0.7

# Annotate negative examples (mark as not visible when ball looks similar to lines)
```

### Issue: Training loss not decreasing

**Causes**:
- Learning rate too high/low
- Bad initialization
- Insufficient data

**Solutions**:
```bash
# Adjust learning rate
python ball_detection/train_tracknet.py --lr 0.00001  # Lower LR

# Check annotations are correct
# Make sure ball positions are accurate

# Add more training data
```

### Issue: Out of memory during training

**Solutions**:
```bash
# Reduce batch size
python ball_detection/train_tracknet.py --batch-size 2

# Use light model
python ball_detection/train_tracknet.py --model light

# Use CPU (slower but no memory limit)
# Model automatically uses CPU if no GPU available
```

## Files

- `tracknet_model.py` - TrackNet architecture (full & light)
- `tracknet_dataset.py` - Dataset and preprocessing
- `annotate_ball.py` - Interactive annotation tool
- `train_tracknet.py` - Training script
- `detect_ball.py` - Inference script

## Next Steps

After ball detection is working:

1. **Ball Tracking** (Phase 4)
   - Temporal smoothing with Kalman filter
   - Handle occlusions and missing detections
   - Predict ball trajectory

2. **Bounce Detection** (Phase 4)
   - Detect when ball hits court
   - Use physics (parabolic trajectory analysis)
   - Combine with court keypoints

3. **In/Out Determination** (Phase 5)
   - Use court homography
   - Project ball position to top-down view
   - Geometric calculation: inside court boundary?

4. **Full Game Analysis** (Phase 6)
   - Rally detection
   - Serve detection
   - Scoring system
   - Match statistics

## References

- **TrackNet Paper**: "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports" (Huang et al., 2019)
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection"

## Citation

If you use this implementation, please cite the original TrackNet paper:

```bibtex
@article{huang2019tracknet,
  title={TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports},
  author={Huang, Yu-Chuan and Liao, I-No and Chen, Ching-Hsuan and {\`I}k, T{\"u}rker and Peng, Wen-Chih},
  journal={arXiv preprint arXiv:1907.03698},
  year={2019}
}
```
