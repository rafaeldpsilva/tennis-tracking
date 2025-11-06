"""
Filter Player Detections to Show Only Tennis Players

Filters out ball boys, umpires, and spectators by:
1. Court region filtering (only players on court)
2. Player count limiting (max 2-4 players in tennis)
3. Position-based filtering (remove people at court edges)
4. Size-based filtering (players are typically larger than ball boys)
"""

import cv2
import numpy as np
import sys
sys.path.append('player_detection')
from detect_players import PlayerDetector


class TennisPlayerFilter:
    def __init__(self, max_players=2, min_player_size=5000):
        self.max_players = max_players
        self.min_player_size = min_player_size
        self.detector = PlayerDetector(confidence_threshold=0.7)

    def create_court_mask(self, frame_shape, court_bounds=None):
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        if court_bounds is None:
            # Default: use center 70% of frame (rough estimate)
            h, w = frame_shape[:2]
            margin_h = int(h * 0.15)
            margin_w = int(w * 0.15)

            # Create polygon for court area
            court_polygon = np.array([
                [margin_w, margin_h],           # top-left
                [w - margin_w, margin_h],       # top-right
                [w - margin_w, h - margin_h],   # bottom-right
                [margin_w, h - margin_h]        # bottom-left
            ], dtype=np.int32)
        else:
            # Use provided court bounds
            court_polygon = np.array([
                court_bounds['top_left'],
                court_bounds['top_right'],
                court_bounds['bottom_right'],
                court_bounds['bottom_left']
            ], dtype=np.int32)

        cv2.fillPoly(mask, [court_polygon], 1)
        return mask

    def filter_by_court_position(self, detections, court_mask, min_overlap=0.5):
        filtered_boxes = []
        filtered_masks = []
        filtered_scores = []

        for box, mask, score in zip(detections['boxes'],
                                     detections['masks'],
                                     detections['scores']):
            # Check if player's feet (bottom of bbox) are on court
            x1, y1, x2, y2 = map(int, box)
            foot_x = (x1 + x2) // 2
            foot_y = y2

            # Check if foot position is on court
            if foot_y < court_mask.shape[0] and foot_x < court_mask.shape[1]:
                if court_mask[foot_y, foot_x] > 0:
                    # Also check overall overlap
                    overlap = np.logical_and(mask, court_mask).sum() / mask.sum()
                    if overlap >= min_overlap:
                        filtered_boxes.append(box)
                        filtered_masks.append(mask)
                        filtered_scores.append(score)

        return {
            'boxes': np.array(filtered_boxes) if filtered_boxes else np.array([]),
            'masks': np.array(filtered_masks) if filtered_masks else np.array([]),
            'scores': np.array(filtered_scores) if filtered_scores else np.array([]),
            'num_players': len(filtered_boxes)
        }

    def filter_by_size(self, detections):
        filtered_boxes = []
        filtered_masks = []
        filtered_scores = []

        for box, mask, score in zip(detections['boxes'],
                                     detections['masks'],
                                     detections['scores']):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            if area >= self.min_player_size:
                filtered_boxes.append(box)
                filtered_masks.append(mask)
                filtered_scores.append(score)

        return {
            'boxes': np.array(filtered_boxes) if filtered_boxes else np.array([]),
            'masks': np.array(filtered_masks) if filtered_masks else np.array([]),
            'scores': np.array(filtered_scores) if filtered_scores else np.array([]),
            'num_players': len(filtered_boxes)
        }

    def filter_by_count(self, detections):
        if detections['num_players'] <= self.max_players:
            return detections

        # Sort by score (descending)
        indices = np.argsort(detections['scores'])[::-1][:self.max_players]

        return {
            'boxes': detections['boxes'][indices],
            'masks': detections['masks'][indices],
            'scores': detections['scores'][indices],
            'num_players': len(indices)
        }

    def filter_players(self, frame, court_bounds=None, use_court_filter=True,
                      use_size_filter=True, use_count_filter=True):
        # Detect all people
        detections = self.detector.detect(frame)

        # Apply filters
        if use_size_filter:
            detections = self.filter_by_size(detections)

        if use_court_filter:
            court_mask = self.create_court_mask(frame.shape, court_bounds)
            detections = self.filter_by_court_position(detections, court_mask, min_overlap=0.3)

        if use_count_filter:
            detections = self.filter_by_count(detections)

        return detections


def process_video_with_filtering(video_path, output_path=None, max_players=4,
                                 court_bounds=None, show_visualization=True):
    """
    Process video and show only tennis players (no ball boys/umpires).

    Args:
        video_path: Input video path
        output_path: Output video path (optional)
        max_players: Maximum number of players to show
        court_bounds: Court boundary (optional, auto-detected if None)
        show_visualization: Show real-time display
    """
    # Initialize filter
    player_filter = TennisPlayerFilter(max_players=max_players, min_player_size=5000)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'='*60}")
    print(f"Tennis Player Filtering")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print(f"Max players: {max_players}")
    print(f"Court filtering: {'Auto' if court_bounds is None else 'Manual'}")
    print(f"{'='*60}\n")

    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detected = 0
    total_filtered = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect and filter players
            detections = player_filter.filter_players(
                frame,
                court_bounds=court_bounds,
                use_court_filter=True,
                use_size_filter=True,
                use_count_filter=True
            )

            # Track statistics
            all_detections = player_filter.detector.detect(frame)
            total_detected += all_detections['num_players']
            total_filtered += detections['num_players']

            # Visualize
            annotated = player_filter.detector.visualize(frame, detections)

            # Add info overlay
            info_text = f"Frame: {frame_count}/{total_frames} | Players: {detections['num_players']}/{all_detections['num_players']}"
            cv2.putText(annotated, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw court boundary (for visualization)
            court_mask = player_filter.create_court_mask(frame.shape, court_bounds)
            contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (0, 255, 255), 2)

            # Write output
            if writer:
                writer.write(annotated)

            # Display
            if show_visualization:
                cv2.imshow('Filtered Player Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress
            if frame_count % 30 == 0:
                print(f"  Processed: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    avg_detected = total_detected / frame_count if frame_count > 0 else 0
    avg_filtered = total_filtered / frame_count if frame_count > 0 else 0

    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Frames processed: {frame_count}")
    print(f"Avg people detected: {avg_detected:.2f}")
    print(f"Avg players shown: {avg_filtered:.2f}")
    print(f"Filter rate: {100*(avg_detected-avg_filtered)/avg_detected:.1f}% removed" if avg_detected > 0 else "")
    if output_path:
        print(f"Output saved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter tennis player detections to remove ball boys and umpires"
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--max-players", type=int, default=2,
                       help="Maximum number of players (2=singles, 4=doubles)")
    parser.add_argument("--min-size", type=int, default=5000,
                       help="Minimum player bounding box area")
    parser.add_argument("--no-viz", action="store_true",
                       help="Don't show visualization window")

    args = parser.parse_args()

    # Process video
    process_video_with_filtering(
        video_path=args.video,
        output_path=args.output,
        max_players=args.max_players,
        show_visualization=not args.no_viz
    )
