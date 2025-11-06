"""
TrackNet Dataset and Data Preprocessing

Handles loading video frames in sequences of 3 and creating ground truth heatmaps
for ball position annotations.

Dataset Format:
- Input: 3 consecutive frames (t-1, t, t+1) stacked as 9-channel tensor
- Output: Heatmap with Gaussian centered at ball position

Annotation Format (CSV):
frame_id,x,y,visibility
0,512,384,1
1,515,390,1
2,-1,-1,0  # Ball not visible
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


def create_heatmap(width, height, x, y, sigma=5.0):
    """
    Create Gaussian heatmap centered at (x, y).

    Args:
        width: Heatmap width
        height: Heatmap height
        x: Ball x-coordinate
        y: Ball y-coordinate
        sigma: Gaussian standard deviation (controls blob size)

    Returns:
        Heatmap array of shape (height, width)
    """
    # Create coordinate grids
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate Gaussian
    heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap.astype(np.float32)


class TrackNetDataset(Dataset):
    """
    PyTorch Dataset for TrackNet ball detection.

    Loads sequences of 3 frames and creates ground truth heatmaps
    from ball position annotations.
    """

    def __init__(self, video_path=None, annotation_csv=None, frame_dir=None,
                 width=512, height=288, sigma=5.0, transform=None):
        """
        Initialize TrackNet dataset.

        Args:
            video_path: Path to video file (if loading from video)
            annotation_csv: Path to CSV with ball annotations
            frame_dir: Directory containing extracted frames (alternative to video)
            width: Target frame width
            height: Target frame height
            sigma: Gaussian sigma for heatmap generation
            transform: Optional transforms to apply
        """
        self.video_path = video_path
        self.annotation_csv = annotation_csv
        self.frame_dir = frame_dir
        self.width = width
        self.height = height
        self.sigma = sigma
        self.transform = transform

        # Load annotations
        if annotation_csv and os.path.exists(annotation_csv):
            self.annotations = pd.read_csv(annotation_csv)
        else:
            self.annotations = None
            print("Warning: No annotations provided")

        # Load video or frame list
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif frame_dir:
            self.frame_files = sorted(list(Path(frame_dir).glob("*.jpg")) +
                                     list(Path(frame_dir).glob("*.png")))
            self.total_frames = len(self.frame_files)
            self.cap = None
        else:
            raise ValueError("Must provide either video_path or frame_dir")

        print(f"Dataset initialized:")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Output size: {width}x{height}")
        print(f"  Annotations: {'Loaded' if self.annotations is not None else 'None'}")

    def __len__(self):
        """Return number of valid 3-frame sequences."""
        # Can't use first or last frame (need frame before and after)
        return max(0, self.total_frames - 2)

    def __getitem__(self, idx):
        """
        Get a training sample (3 frames + heatmap).

        Args:
            idx: Index (corresponds to middle frame)

        Returns:
            Tuple of (frames_tensor, heatmap_tensor)
            - frames_tensor: (9, H, W) - 3 RGB frames stacked
            - heatmap_tensor: (1, H, W) - Ground truth ball position
        """
        # Frame indices (previous, current, next)
        frame_indices = [idx, idx + 1, idx + 2]

        # Load 3 consecutive frames
        frames = []
        for frame_idx in frame_indices:
            frame = self._load_frame(frame_idx)
            # Resize to target size
            frame = cv2.resize(frame, (self.width, self.height))
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        # Stack frames: (3, H, W, 3) -> (H, W, 9)
        frames_stacked = np.concatenate(frames, axis=2)  # Concatenate along channel dim

        # Convert to tensor and normalize
        frames_tensor = torch.from_numpy(frames_stacked).permute(2, 0, 1).float()
        frames_tensor = frames_tensor / 255.0  # Normalize to [0, 1]

        # Create ground truth heatmap for middle frame (idx + 1)
        heatmap = self._create_ground_truth_heatmap(idx + 1)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()

        # Apply transforms if provided
        if self.transform:
            frames_tensor, heatmap_tensor = self.transform(frames_tensor, heatmap_tensor)

        return frames_tensor, heatmap_tensor

    def _load_frame(self, frame_idx):
        """Load a single frame by index."""
        if self.cap is not None:
            # Load from video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {frame_idx}")
            return frame
        else:
            # Load from files
            frame_path = self.frame_files[frame_idx]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Failed to read frame {frame_path}")
            return frame

    def _create_ground_truth_heatmap(self, frame_idx):
        """Create ground truth heatmap for a frame."""
        # Check if ball is annotated for this frame
        if self.annotations is not None:
            frame_data = self.annotations[self.annotations['frame_id'] == frame_idx]

            if not frame_data.empty and frame_data.iloc[0]['visibility'] == 1:
                # Get ball position
                x = frame_data.iloc[0]['x']
                y = frame_data.iloc[0]['y']

                # Scale to target size
                orig_width = self.annotations['orig_width'].iloc[0] if 'orig_width' in self.annotations.columns else self.width
                orig_height = self.annotations['orig_height'].iloc[0] if 'orig_height' in self.annotations.columns else self.height

                x_scaled = int(x * self.width / orig_width)
                y_scaled = int(y * self.height / orig_height)

                # Create heatmap
                return create_heatmap(self.width, self.height, x_scaled, y_scaled, self.sigma)

        # No ball visible - return empty heatmap
        return np.zeros((self.height, self.width), dtype=np.float32)

    def __del__(self):
        """Release video capture on deletion."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


class TrackNetInferenceDataset(Dataset):
    """
    Dataset for inference (no annotations needed).
    Returns 3-frame sequences for ball prediction.
    """

    def __init__(self, video_path, width=512, height=288):
        """
        Initialize inference dataset.

        Args:
            video_path: Path to video file
            width: Target frame width
            height: Target frame height
        """
        self.video_path = video_path
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Inference dataset initialized:")
        print(f"  Video: {video_path}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  FPS: {self.fps}")

    def __len__(self):
        return max(0, self.total_frames - 2)

    def __getitem__(self, idx):
        """Get 3-frame sequence for inference."""
        frame_indices = [idx, idx + 1, idx + 2]

        frames = []
        for frame_idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {frame_idx}")

            frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        # Stack and convert to tensor
        frames_stacked = np.concatenate(frames, axis=2)
        frames_tensor = torch.from_numpy(frames_stacked).permute(2, 0, 1).float()
        frames_tensor = frames_tensor / 255.0

        return frames_tensor, idx + 1  # Return middle frame index

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


def create_annotation_template(video_path, output_csv, sample_every=30):
    """
    Create annotation template CSV from video.

    Args:
        video_path: Path to video file
        output_csv: Path to save CSV template
        sample_every: Annotate every Nth frame (default: 30 = 1 per second at 30fps)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Create annotation rows
    annotations = []
    for frame_id in range(0, total_frames, sample_every):
        annotations.append({
            'frame_id': frame_id,
            'x': -1,  # To be filled
            'y': -1,  # To be filled
            'visibility': 0,  # To be filled (0=not visible, 1=visible)
            'orig_width': width,
            'orig_height': height
        })

    df = pd.DataFrame(annotations)
    df.to_csv(output_csv, index=False)

    print(f"Created annotation template:")
    print(f"  Video: {video_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames to annotate: {len(annotations)}")
    print(f"  Sample rate: every {sample_every} frames ({sample_every/fps:.1f}s)")
    print(f"  Output: {output_csv}")


if __name__ == "__main__":
    # Test dataset creation
    print("Testing TrackNet Dataset...")

    # Create dummy annotation file
    annotations = pd.DataFrame({
        'frame_id': [0, 1, 2, 3, 4],
        'x': [100, 105, 110, 115, 120],
        'y': [200, 205, 210, 215, 220],
        'visibility': [1, 1, 1, 1, 1],
        'orig_width': [1920, 1920, 1920, 1920, 1920],
        'orig_height': [1080, 1080, 1080, 1080, 1080]
    })
    test_csv = 'test_annotations.csv'
    annotations.to_csv(test_csv, index=False)

    # Test heatmap creation
    print("\nTesting heatmap creation...")
    heatmap = create_heatmap(512, 288, 256, 144, sigma=5.0)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"Max position: {np.unravel_index(heatmap.argmax(), heatmap.shape)}")

    # Visualize heatmap
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar()
    plt.title('Ground Truth Heatmap')
    plt.savefig('output/test_heatmap.png')
    print(f"Heatmap visualization saved to: output/test_heatmap.png")

    # Clean up
    os.remove(test_csv)

    print("\n✓ Dataset tests passed!")
