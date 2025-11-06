"""
TrackNet Ball Detection Inference

Run trained TrackNet model on tennis videos to detect ball positions.

Usage:
    python detect_ball.py --video input.mp4 --model best_model.pth --output output.mp4
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from tracknet_model import TrackNet, TrackNetLightning
from tracknet_dataset import TrackNetInferenceDataset


class BallDetector:
    """
    Tennis ball detector using TrackNet.
    """

    def __init__(self, model_path, model_type='full', device=None, threshold=0.5):
        """
        Initialize ball detector.

        Args:
            model_path: Path to trained model weights
            model_type: 'full' or 'light'
            device: torch device (auto-detect if None)
            threshold: Detection confidence threshold
        """
        self.model_path = model_path
        self.model_type = model_type
        self.threshold = threshold

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model
        print(f"Loading {model_type} TrackNet model from {model_path}")
        if model_type == 'light':
            self.model = TrackNetLightning()
        else:
            self.model = TrackNet()

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_accuracy' in checkpoint:
                print(f"Model validation accuracy: {checkpoint['val_accuracy']:.2%}")
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        print("✓ Model loaded successfully")

    def detect(self, frames_tensor):
        """
        Detect ball in a 3-frame sequence.

        Args:
            frames_tensor: (9, H, W) tensor of 3 stacked RGB frames

        Returns:
            Tuple of (x, y, confidence) or None if no ball detected
        """
        # Add batch dimension and move to device
        if frames_tensor.dim() == 3:
            frames_tensor = frames_tensor.unsqueeze(0)

        frames_tensor = frames_tensor.to(self.device)

        # Run model
        with torch.no_grad():
            heatmap = self.model(frames_tensor)

        # Extract ball position
        position = self.model.predict_ball_position(heatmap, threshold=self.threshold)

        return position

    def process_video(self, video_path, output_path=None, show_visualization=True,
                     draw_trajectory=True, trajectory_length=30):
        """
        Process video and detect ball in each frame.

        Args:
            video_path: Input video path
            output_path: Output video path (optional)
            show_visualization: Show real-time visualization
            draw_trajectory: Draw ball trajectory
            trajectory_length: Number of previous positions to show
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps} FPS")
        print(f"Total frames: {total_frames}")
        print(f"{'='*60}\n")

        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Ball trajectory
        trajectory = []

        # Create inference dataset
        dataset = TrackNetInferenceDataset(video_path, width=512, height=288)

        # Process frames
        detections = []
        for frames_tensor, frame_idx in tqdm(dataset, desc="Detecting ball"):
            # Detect ball
            position = self.detect(frames_tensor)

            if position is not None:
                x, y, conf = position
                # Scale back to original resolution
                x_orig = int(x * width / 512)
                y_orig = int(y * height / 288)
                detections.append({'frame': frame_idx, 'x': x_orig, 'y': y_orig, 'conf': conf})
                trajectory.append((x_orig, y_orig))
            else:
                detections.append({'frame': frame_idx, 'x': -1, 'y': -1, 'conf': 0})
                trajectory.append(None)

            # Limit trajectory length
            if len(trajectory) > trajectory_length:
                trajectory.pop(0)

            # Visualize
            if output_path or show_visualization:
                # Read original frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Draw ball position
                if position is not None:
                    # Draw circle
                    cv2.circle(frame, (x_orig, y_orig), 10, (0, 255, 0), 2)
                    cv2.circle(frame, (x_orig, y_orig), 3, (0, 255, 0), -1)

                    # Draw confidence
                    label = f"Ball: {conf:.2f}"
                    cv2.putText(frame, label, (x_orig + 15, y_orig - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw trajectory
                if draw_trajectory:
                    for i in range(len(trajectory) - 1):
                        if trajectory[i] is not None and trajectory[i+1] is not None:
                            cv2.line(frame, trajectory[i], trajectory[i+1],
                                   (255, 0, 0), 2)

                # Draw frame info
                info_text = f"Frame: {frame_idx}/{total_frames}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Write to output
                if writer:
                    writer.write(frame)

                # Display
                if show_visualization:
                    cv2.imshow('Ball Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Statistics
        detected_frames = sum(1 for d in detections if d['x'] >= 0)
        detection_rate = detected_frames / len(detections) if len(detections) > 0 else 0

        print(f"\n{'='*60}")
        print(f"Detection complete!")
        print(f"{'='*60}")
        print(f"Total frames processed: {len(detections)}")
        print(f"Ball detected in: {detected_frames} frames ({detection_rate:.1%})")
        if output_path:
            print(f"Output saved to: {output_path}")
        print(f"{'='*60}")

        return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect tennis ball using TrackNet")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--model-type", type=str, default='full', choices=['full', 'light'],
                       help="Model type")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Detection confidence threshold")
    parser.add_argument("--no-viz", action="store_true", help="Don't show visualization")
    parser.add_argument("--no-trajectory", action="store_true",
                       help="Don't draw ball trajectory")
    parser.add_argument("--trajectory-length", type=int, default=30,
                       help="Number of trajectory points to show")

    args = parser.parse_args()

    # Create detector
    detector = BallDetector(
        model_path=args.model,
        model_type=args.model_type,
        threshold=args.threshold
    )

    # Process video
    detector.process_video(
        video_path=args.video,
        output_path=args.output,
        show_visualization=not args.no_viz,
        draw_trajectory=not args.no_trajectory,
        trajectory_length=args.trajectory_length
    )
