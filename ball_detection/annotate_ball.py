"""
Interactive Ball Annotation Tool

GUI tool for annotating tennis ball positions in video frames.
Click on the ball to mark its position, press 'v' to mark as not visible.

Controls:
- Click: Mark ball position
- 'v': Mark ball as not visible
- 'n': Next frame
- 'b': Previous frame
- 's': Save annotations
- 'q': Quit
- Space: Next frame (same as 'n')
- Backspace: Previous frame (same as 'b')
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


class BallAnnotator:
    """
    Interactive tool for annotating ball positions in tennis videos.
    """

    def __init__(self, video_path, output_csv, start_frame=0, sample_rate=1):
        """
        Initialize ball annotator.

        Args:
            video_path: Path to video file
            output_csv: Path to save annotations
            start_frame: Frame to start annotation from
            sample_rate: Annotate every Nth frame (1 = every frame)
        """
        self.video_path = video_path
        self.output_csv = output_csv
        self.sample_rate = sample_rate

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Current state
        self.current_frame_idx = start_frame
        self.current_frame = None
        self.annotations = {}

        # Load existing annotations if available
        if Path(output_csv).exists():
            self._load_annotations()

        # Mouse callback state
        self.ball_position = None

        print(f"\n{'='*60}")
        print(f"Ball Annotation Tool")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Sample rate: Every {sample_rate} frame(s)")
        print(f"Output: {output_csv}")
        print(f"{'='*60}\n")

        print("Controls:")
        print("  Click: Mark ball position")
        print("  'v': Mark as not visible")
        print("  'n' or Space: Next frame")
        print("  'b' or Backspace: Previous frame")
        print("  's': Save annotations")
        print("  'q' or ESC: Save and quit")
        print(f"{'='*60}\n")

    def _load_annotations(self):
        """Load existing annotations from CSV."""
        try:
            df = pd.read_csv(self.output_csv)
            for _, row in df.iterrows():
                frame_id = int(row['frame_id'])
                self.annotations[frame_id] = {
                    'x': int(row['x']) if row['x'] >= 0 else -1,
                    'y': int(row['y']) if row['y'] >= 0 else -1,
                    'visibility': int(row['visibility'])
                }
            print(f"Loaded {len(self.annotations)} existing annotations")
        except Exception as e:
            print(f"Could not load existing annotations: {e}")

    def _save_annotations(self):
        """Save annotations to CSV."""
        rows = []
        for frame_id, data in sorted(self.annotations.items()):
            rows.append({
                'frame_id': frame_id,
                'x': data['x'],
                'y': data['y'],
                'visibility': data['visibility'],
                'orig_width': self.width,
                'orig_height': self.height
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_csv, index=False)
        print(f"Saved {len(rows)} annotations to {self.output_csv}")

    def _load_frame(self, frame_idx):
        """Load a specific frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to mark ball position."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ball_position = (x, y)
            self.annotations[self.current_frame_idx] = {
                'x': x,
                'y': y,
                'visibility': 1
            }
            print(f"Frame {self.current_frame_idx}: Ball at ({x}, {y})")
            self._redraw()

    def _redraw(self):
        """Redraw current frame with annotations."""
        if self.current_frame is None:
            return

        display = self.current_frame.copy()

        # Draw current annotation if exists
        if self.current_frame_idx in self.annotations:
            ann = self.annotations[self.current_frame_idx]
            if ann['visibility'] == 1:
                # Draw circle at ball position
                cv2.circle(display, (ann['x'], ann['y']), 10, (0, 255, 0), 2)
                cv2.circle(display, (ann['x'], ann['y']), 3, (0, 255, 0), -1)

                # Draw label
                label = f"Ball: ({ann['x']}, {ann['y']})"
                cv2.putText(display, label, (ann['x'] + 15, ann['y'] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Mark as not visible
                cv2.putText(display, "Ball: NOT VISIBLE", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw frame info
        info_text = f"Frame: {self.current_frame_idx}/{self.total_frames-1} | Annotated: {len(self.annotations)}"
        cv2.putText(display, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw progress bar
        progress = self.current_frame_idx / self.total_frames
        bar_width = self.width - 20
        cv2.rectangle(display, (10, self.height - 30), (10 + int(bar_width * progress), self.height - 20),
                     (0, 255, 0), -1)
        cv2.rectangle(display, (10, self.height - 30), (10 + bar_width, self.height - 20),
                     (255, 255, 255), 2)

        cv2.imshow('Ball Annotation', display)

    def _next_frame(self):
        """Go to next frame."""
        next_idx = self.current_frame_idx + self.sample_rate
        if next_idx < self.total_frames:
            self.current_frame_idx = next_idx
            self.current_frame = self._load_frame(self.current_frame_idx)
            self.ball_position = None
            self._redraw()

    def _previous_frame(self):
        """Go to previous frame."""
        prev_idx = self.current_frame_idx - self.sample_rate
        if prev_idx >= 0:
            self.current_frame_idx = prev_idx
            self.current_frame = self._load_frame(self.current_frame_idx)
            self.ball_position = None
            self._redraw()

    def _mark_not_visible(self):
        """Mark current frame as ball not visible."""
        self.annotations[self.current_frame_idx] = {
            'x': -1,
            'y': -1,
            'visibility': 0
        }
        print(f"Frame {self.current_frame_idx}: Ball not visible")
        self._redraw()

    def run(self):
        """Run the annotation tool."""
        # Create window
        cv2.namedWindow('Ball Annotation')
        cv2.setMouseCallback('Ball Annotation', self._mouse_callback)

        # Load first frame
        self.current_frame = self._load_frame(self.current_frame_idx)
        self._redraw()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                self._save_annotations()
                break
            elif key == ord('n') or key == ord(' '):  # Next frame
                self._next_frame()
            elif key == ord('b') or key == 8:  # Previous frame (backspace)
                self._previous_frame()
            elif key == ord('v'):  # Mark as not visible
                self._mark_not_visible()
            elif key == ord('s'):  # Save
                self._save_annotations()

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()


def batch_annotate(video_dir, output_dir, sample_rate=30):
    """
    Create annotation templates for multiple videos.

    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save annotation CSV files
        sample_rate: Annotate every Nth frame
    """
    from ball_detection.tracknet_dataset import create_annotation_template

    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))

    print(f"Found {len(video_files)} videos")

    for video_path in video_files:
        output_csv = output_dir / f"{video_path.stem}_annotations.csv"
        print(f"\nProcessing: {video_path.name}")
        create_annotation_template(str(video_path), str(output_csv), sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate tennis ball positions")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, help="Path to save annotations CSV")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="Frame to start annotation from")
    parser.add_argument("--sample-rate", type=int, default=1,
                       help="Annotate every Nth frame (default: 1 = every frame)")
    parser.add_argument("--batch", type=str, help="Batch create templates for videos in directory")
    parser.add_argument("--batch-output", type=str, help="Output directory for batch templates")
    parser.add_argument("--batch-sample", type=int, default=30,
                       help="Sample rate for batch template creation")

    args = parser.parse_args()

    if args.batch:
        # Batch template creation
        output_dir = args.batch_output or "ball_annotations"
        batch_annotate(args.batch, output_dir, args.batch_sample)
    elif args.video and args.output:
        # Interactive annotation
        annotator = BallAnnotator(
            video_path=args.video,
            output_csv=args.output,
            start_frame=args.start_frame,
            sample_rate=args.sample_rate
        )
        annotator.run()
    else:
        print("Usage:")
        print("  Interactive annotation:")
        print("    python annotate_ball.py --video input.mp4 --output annotations.csv")
        print("  Batch template creation:")
        print("    python annotate_ball.py --batch VideoInput/ --batch-output ball_annotations/")
