"""
Combined Court + Player Detection and Analysis

Integrates court keypoint detection with player detection to provide:
- Court boundary visualization
- Player detection and segmentation
- Player positioning relative to court
- Filtering to show only players on the court

This is the foundation for ball tracking, bounce detection, and in/out calls.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Import our custom modules
from detect_players import PlayerDetector


class TennisAnalyzer:
    """
    Complete tennis court and player analysis system.

    Combines court detection, player detection, and geometric analysis
    to provide real-time tennis match insights.
    """

    def __init__(self, player_confidence=0.7, court_detector=None):
        """
        Initialize the tennis analyzer.

        Args:
            player_confidence: Confidence threshold for player detection
            court_detector: Court keypoint detector (optional, for future integration)
        """
        self.player_detector = PlayerDetector(confidence_threshold=player_confidence)
        self.court_detector = court_detector

        # Court state
        self.court_keypoints = None
        self.court_homography = None

        print("✓ Tennis Analyzer initialized")

    def set_court_keypoints(self, keypoints):
        """
        Set detected court keypoints.

        Args:
            keypoints: Dict mapping keypoint names to (x, y) coordinates
                      e.g., {'top_left_baseline': (x, y), ...}
        """
        self.court_keypoints = keypoints
        print(f"✓ Court keypoints set: {len(keypoints)} points")

    def set_court_homography(self, homography_matrix):
        """
        Set homography matrix for court perspective transformation.

        Args:
            homography_matrix: 3x3 transformation matrix
        """
        self.court_homography = homography_matrix
        print("✓ Court homography set")

    def create_court_mask(self, frame_shape, court_keypoints=None):
        """
        Create binary mask of court region.

        Args:
            frame_shape: (height, width) of frame
            court_keypoints: Dict of keypoint coordinates (uses self.court_keypoints if None)

        Returns:
            Binary mask (numpy array) with 1s inside court, 0s outside
        """
        if court_keypoints is None:
            court_keypoints = self.court_keypoints

        if court_keypoints is None:
            # No court detected, return full frame mask
            return np.ones(frame_shape[:2], dtype=np.uint8)

        # Create mask
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        # Define court boundary (convex hull of all keypoints)
        points = np.array([list(pt) for pt in court_keypoints.values()], dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 1)

        return mask

    def detect_players_on_court(self, frame, min_overlap=0.3):
        """
        Detect players and filter to only those on the court.

        Args:
            frame: BGR image
            min_overlap: Minimum overlap with court region to be considered "on court"

        Returns:
            Same format as PlayerDetector.detect()
        """
        # Get court mask
        court_mask = self.create_court_mask(frame.shape, self.court_keypoints)

        # Detect and filter players
        detections = self.player_detector.detect_and_filter_by_court(
            frame, court_mask, min_overlap
        )

        return detections

    def draw_court_lines(self, frame, keypoints=None, color=(0, 255, 255),
                        thickness=2):
        """
        Draw court lines on frame.

        Args:
            frame: BGR image
            keypoints: Dict of court keypoints (uses self.court_keypoints if None)
            color: Line color (B, G, R)
            thickness: Line thickness

        Returns:
            Frame with court lines drawn
        """
        if keypoints is None:
            keypoints = self.court_keypoints

        if keypoints is None:
            return frame

        annotated = frame.copy()

        # Define line connections based on tennis court structure
        # This is a simple version - you'll expand this with the 14-keypoint schema
        line_connections = [
            ('top_left_baseline', 'top_right_baseline'),
            ('bottom_left_baseline', 'bottom_right_baseline'),
            ('top_left_baseline', 'bottom_left_baseline'),
            ('top_right_baseline', 'bottom_right_baseline'),
        ]

        for pt1_name, pt2_name in line_connections:
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = tuple(map(int, keypoints[pt1_name]))
                pt2 = tuple(map(int, keypoints[pt2_name]))
                cv2.line(annotated, pt1, pt2, color, thickness)

        # Draw keypoints
        for name, (x, y) in keypoints.items():
            cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)

        return annotated

    def analyze_frame(self, frame, draw_court=True, draw_players=True,
                     filter_by_court=True):
        """
        Complete frame analysis: detect court + players.

        Args:
            frame: BGR image
            draw_court: Draw court lines
            draw_players: Draw player detections
            filter_by_court: Only show players on court

        Returns:
            Tuple of (annotated_frame, analysis_dict)
        """
        annotated = frame.copy()
        analysis = {}

        # Detect players
        if filter_by_court and self.court_keypoints is not None:
            player_detections = self.detect_players_on_court(frame)
            analysis['players_filtered'] = True
        else:
            player_detections = self.player_detector.detect(frame)
            analysis['players_filtered'] = False

        analysis['num_players'] = player_detections['num_players']
        analysis['player_positions'] = self.player_detector.get_player_foot_positions(
            player_detections
        )

        # Draw court
        if draw_court and self.court_keypoints is not None:
            annotated = self.draw_court_lines(annotated)

        # Draw players
        if draw_players:
            annotated = self.player_detector.visualize(annotated, player_detections)

        return annotated, analysis

    def process_video(self, video_path, output_path=None, show_visualization=True,
                     filter_by_court=True):
        """
        Process video with court + player analysis.

        Args:
            video_path: Input video path
            output_path: Output video path (optional)
            show_visualization: Show real-time display
            filter_by_court: Filter players to only those on court
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps} FPS")
        print(f"Total frames: {total_frames}")
        print(f"Court filtering: {'Enabled' if filter_by_court else 'Disabled'}")
        print(f"{'='*60}\n")

        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Analyze frame
                annotated, analysis = self.analyze_frame(
                    frame,
                    draw_court=True,
                    draw_players=True,
                    filter_by_court=filter_by_court
                )

                # Add info overlay
                info_lines = [
                    f"Frame: {frame_count}/{total_frames}",
                    f"Players: {analysis['num_players']}",
                    f"Filtered: {'Yes' if analysis['players_filtered'] else 'No'}",
                ]

                y_offset = 30
                for line in info_lines:
                    cv2.putText(annotated, line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_offset += 25

                # Write output
                if writer:
                    writer.write(annotated)

                # Display
                if show_visualization:
                    cv2.imshow('Tennis Analysis', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Progress
                if frame_count % 30 == 0:
                    progress = 100 * frame_count / total_frames
                    print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)")

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        print(f"\n✓ Processing complete: {frame_count} frames")
        if output_path:
            print(f"✓ Output saved: {output_path}")


def demo_with_manual_court():
    """
    Demo with manually specified court keypoints.
    Use this until court keypoint detection is integrated.
    """
    print("\n" + "="*60)
    print("TENNIS COURT + PLAYER ANALYSIS DEMO")
    print("="*60)

    # Initialize analyzer
    analyzer = TennisAnalyzer(player_confidence=0.6)

    # Manually set court keypoints (example coordinates for video_input2.mp4)
    # You can adjust these based on your specific video
    example_keypoints = {
        'top_left_baseline': (182, 429),
        'top_right_baseline': (783, 428),
        'bottom_left_baseline': (288, 152),
        'bottom_right_baseline': (668, 150),
    }

    analyzer.set_court_keypoints(example_keypoints)

    # Process video
    video_path = "VideoInput/video_input6.mp4"
    output_path = "output/court_player_analysis.mp4"

    Path("output").mkdir(exist_ok=True)

    analyzer.process_video(
        video_path=video_path,
        output_path=output_path,
        show_visualization=False,  # Set to True to see real-time display
        filter_by_court=False  # Set to True once court coordinates are accurate
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combined court and player detection for tennis analysis"
    )
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--confidence", type=float, default=0.6,
                       help="Player detection confidence threshold")
    parser.add_argument("--filter-court", action="store_true",
                       help="Filter players to only those on court")

    args = parser.parse_args()

    if args.demo or not args.video:
        demo_with_manual_court()
    else:
        analyzer = TennisAnalyzer(player_confidence=args.confidence)
        analyzer.process_video(
            video_path=args.video,
            output_path=args.output,
            show_visualization=True,
            filter_by_court=args.filter_court
        )
