import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import torch


class PlayerDetector:
    def __init__(self, confidence_threshold=0.7, model_type="R50-FPN"):
        self.confidence_threshold = confidence_threshold
        self.cfg = self._setup_config(model_type, confidence_threshold)
        self.predictor = DefaultPredictor(self.cfg)

        self.PERSON_CLASS_ID = 0

        print(f"✓ Mask R-CNN player detector initialized")
        print(f"  Model: {model_type}")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Device: {self.cfg.MODEL.DEVICE}")

    def _setup_config(self, model_type, confidence_threshold):
        cfg = get_cfg()

        # Select pre-trained model
        model_zoo_configs = {
            "R50-FPN": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "R101-FPN": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            "X101-FPN": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        }

        config_file = model_zoo_configs.get(model_type, model_zoo_configs[model_type])
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold

        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        return cfg

    def detect(self, frame):
        # Run inference
        outputs = self.predictor(frame)

        # Extract predictions
        instances = outputs["instances"].to("cpu")

        # Filter to only 'person' class
        person_indices = instances.pred_classes == self.PERSON_CLASS_ID

        # Extract person detections
        boxes = instances.pred_boxes[person_indices].tensor.numpy()
        masks = instances.pred_masks[person_indices].numpy()
        scores = instances.scores[person_indices].numpy()

        return {
            'boxes': boxes,
            'masks': masks,
            'scores': scores,
            'num_players': len(boxes)
        }

    def detect_and_filter_by_court(self, frame, court_mask=None, min_overlap=0.3):
        detections = self.detect(frame)

        if court_mask is None:
            return detections

        filtered_boxes = []
        filtered_masks = []
        filtered_scores = []

        for box, mask, score in zip(detections['boxes'],
                                     detections['masks'],
                                     detections['scores']):
            overlap = np.logical_and(mask, court_mask).sum() / mask.sum()

            if overlap >= min_overlap:
                filtered_boxes.append(box)
                filtered_masks.append(mask)
                filtered_scores.append(score)

        return {
            'boxes': np.array(filtered_boxes),
            'masks': np.array(filtered_masks),
            'scores': np.array(filtered_scores),
            'num_players': len(filtered_boxes)
        }

    def visualize(self, frame, detections, show_masks=True, show_boxes=True,
                  show_labels=True):
        annotated = frame.copy()

        for i, (box, mask, score) in enumerate(zip(detections['boxes'],
                                                     detections['masks'],
                                                     detections['scores'])):
            # Draw mask
            if show_masks and len(mask) > 0:
                color = np.random.randint(0, 255, 3).tolist()
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask] = color
                annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.5, 0)

            # Draw bounding box
            if show_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                if show_labels:
                    label = f"Player {i+1}: {score:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated

    def get_player_centers(self, detections):
        centers = []
        for box in detections['boxes']:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))

        return centers

    def get_player_foot_positions(self, detections):
        foot_positions = []
        for box in detections['boxes']:
            x1, y1, x2, y2 = box
            foot_x = (x1 + x2) / 2
            foot_y = y2  # Bottom of bounding box
            foot_positions.append((foot_x, foot_y))

        return foot_positions


def process_video(video_path, output_path=None, confidence_threshold=0.7, model_type="R50-FPN",
                  show_visualization=True):
    # Initialize detector
    detector = PlayerDetector(confidence_threshold=confidence_threshold, model_type=model_type)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_players_detected = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect players
            detections = detector.detect(frame)
            total_players_detected += detections['num_players']

            # Visualize
            annotated = detector.visualize(frame, detections)

            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Players: {detections['num_players']}"
            cv2.putText(annotated, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Write to output
            if writer:
                writer.write(annotated)

            # Display
            if show_visualization:
                cv2.imshow('Player Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames "
                      f"({100*frame_count/total_frames:.1f}%)")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    avg_players = total_players_detected / frame_count if frame_count > 0 else 0
    print(f"\n✓ Processing complete!")
    print(f"  Frames processed: {frame_count}")
    print(f"  Average players per frame: {avg_players:.2f}")
    if output_path:
        print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect tennis players using Mask R-CNN")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, help="Path to output video")
    parser.add_argument("--confidence", type=float, default=0.7,
                       help="Confidence threshold (default: 0.7)")
    parser.add_argument("--image", type=str, help="Path to single image (for testing)")
    parser.add_argument("--model", type=str, default="R50-FPN",
                       choices=["R50-FPN", "R101-FPN", "X101-FPN"],
                       help="Mask R-CNN model architecture")

    args = parser.parse_args()

    if args.image:
        # Test on single image
        print(f"Testing on image: {args.image}")
        detector = PlayerDetector(confidence_threshold=args.confidence,
                                 model_type=args.model)

        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            exit(1)

        detections = detector.detect(image)
        print(f"Detected {detections['num_players']} players")

        annotated = detector.visualize(image, detections)
        cv2.imshow('Player Detection', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if args.output:
            cv2.imwrite(args.output, annotated)
            print(f"Saved to: {args.output}")

    elif args.video:
        # Process video
        process_video(args.video, args.output, args.confidence, args.model)

    else:
        print("Please provide either --video or --image argument")
        print("Example: python detect_players.py --video VideoInput/video_input2.mp4")
