"""
Test trained Keypoint R-CNN model on new images.

This script helps you:
1. Load your trained model
2. Run inference on test images
3. Visualize predictions with measurements
4. Diagnose specific issues like baseline width bias
"""

import cv2
import numpy as np
from pathlib import Path
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def setup_predictor(model_path, num_keypoints=8):
    """
    Load trained model for inference.

    Args:
        model_path: Path to model_final.pth
        num_keypoints: Number of keypoints (8 for tennis court)
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

    # Model configuration
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"  # Change to "cuda" if running on Colab

    # Set metadata for visualization
    MetadataCatalog.get("tennis_court_test").set(
        thing_classes=["tennis_court"],
        keypoint_names=[
            "top_left_baseline",
            "top_right_baseline",
            "bottom_left_baseline",
            "bottom_right_baseline",
            "top_left_service",
            "top_right_service",
            "bottom_left_service",
            "bottom_right_service",
        ],
        keypoint_connection_rules=[
            ("top_left_baseline", "top_right_baseline", (255, 0, 0)),
            ("top_right_baseline", "bottom_right_baseline", (0, 255, 0)),
            ("bottom_right_baseline", "bottom_left_baseline", (255, 0, 0)),
            ("bottom_left_baseline", "top_left_baseline", (0, 255, 0)),
            ("top_left_service", "top_right_service", (0, 0, 255)),
            ("bottom_left_service", "bottom_right_service", (0, 0, 255)),
            ("top_left_service", "bottom_left_service", (255, 255, 0)),
            ("top_right_service", "bottom_right_service", (255, 255, 0)),
        ]
    )

    predictor = DefaultPredictor(cfg)
    return predictor


def measure_baseline_widths(keypoints):
    """
    Measure top and bottom baseline widths from predicted keypoints.

    Returns dict with measurements and analysis.
    """
    # Extract keypoint coordinates
    kps = keypoints.cpu().numpy()[0]  # [num_keypoints, 3] (x, y, confidence)

    top_left = kps[0, :2]
    top_right = kps[1, :2]
    bottom_left = kps[2, :2]
    bottom_right = kps[3, :2]

    # Calculate widths
    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)

    # Perspective ratio (bottom should be wider due to camera angle)
    ratio = bottom_width / top_width if top_width > 0 else 0

    # Confidence scores
    top_conf = (kps[0, 2] + kps[1, 2]) / 2
    bottom_conf = (kps[2, 2] + kps[3, 2]) / 2

    return {
        'top_width': top_width,
        'bottom_width': bottom_width,
        'ratio': ratio,
        'width_diff': bottom_width - top_width,
        'top_confidence': top_conf,
        'bottom_confidence': bottom_conf,
    }


def visualize_with_measurements(image, outputs, save_path=None):
    """
    Visualize predictions with baseline width measurements overlaid.
    """
    metadata = MetadataCatalog.get("tennis_court_test")

    # Standard visualization
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = out.get_image()[:, :, ::-1]

    # Get measurements
    if len(outputs["instances"]) > 0:
        keypoints = outputs["instances"].pred_keypoints
        measurements = measure_baseline_widths(keypoints)

        # Draw measurements on image
        kps = keypoints.cpu().numpy()[0]

        # Top baseline width
        top_left = tuple(kps[0, :2].astype(int))
        top_right = tuple(kps[1, :2].astype(int))
        mid_top = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)

        """ cv2.putText(result, f"Top: {measurements['top_width']:.0f}px",(mid_top[0] - 50, mid_top[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Bottom baseline width
        bottom_left = tuple(kps[2, :2].astype(int))
        bottom_right = tuple(kps[3, :2].astype(int))
        mid_bottom = ((bottom_left[0] + bottom_right[0]) // 2, (bottom_left[1] + bottom_right[1]) // 2)

        cv2.putText(result, f"Bottom: {measurements['bottom_width']:.0f}px",
                   (mid_bottom[0] - 50, mid_bottom[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Ratio
        cv2.putText(result, f"Ratio: {measurements['ratio']:.2f}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Confidence
        cv2.putText(result, f"Top conf: {measurements['top_confidence']:.2f}",
                   (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Bottom conf: {measurements['bottom_confidence']:.2f}",
                   (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 """
        # Print analysis
        print(f"\n📊 Baseline Measurements:")
        print(f"  Top baseline width: {measurements['top_width']:.1f} pixels")
        print(f"  Bottom baseline width: {measurements['bottom_width']:.1f} pixels")
        print(f"  Width difference: {measurements['width_diff']:.1f} pixels")
        print(f"  Bottom/Top ratio: {measurements['ratio']:.2f}")
        print(f"  Top confidence: {measurements['top_confidence']:.2f}")
        print(f"  Bottom confidence: {measurements['bottom_confidence']:.2f}")

        # Diagnosis
        if measurements['ratio'] > 1.3:
            print("\n⚠️  Bottom baseline is significantly wider than top!")
            print("   This might indicate:")
            print("   - Normal perspective (bottom closer to camera)")
            print("   - OR model bias (predicting too wide)")
        elif measurements['bottom_confidence'] < measurements['top_confidence']:
            print("\n⚠️  Bottom keypoints have lower confidence")
            print("   Model is less certain about bottom corners")

    if save_path:
        cv2.imwrite(save_path, result)
        print(f"\n💾 Saved visualization to: {save_path}")

    return result, measurements if len(outputs["instances"]) > 0 else None


def main():
    """
    Test model on sample images.
    """
    print("="*60)
    print("Tennis Court Keypoint Detection - Model Testing")
    print("="*60)

    # Configuration
    model_path = "output/keypoint_rcnn/model_final.pth"
    test_images_dir = "training_data/frames"  # Or use new test images

    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("Please download model_final.pth from Colab and place it in output/keypoint_rcnn/")
        return

    print(f"\n[1/3] Loading model from: {model_path}")
    predictor = setup_predictor(model_path)
    print("  ✓ Model loaded successfully")

    # Get test images
    test_images = list(Path(test_images_dir).glob("*.jpg"))
    if not test_images:
        print(f"\n❌ No images found in: {test_images_dir}")
        return

    print(f"\n[2/3] Found {len(test_images)} test images")

    # Test on first 3 images
    output_dir = Path("output/test_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_measurements = []

    for i, img_path in enumerate(test_images[:3], 1):
        print(f"\n[3/{3+i-1}] Testing on: {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))

        # Run inference
        outputs = predictor(img)

        # Visualize with measurements
        save_path = output_dir / f"prediction_{img_path.stem}.jpg"
        result, measurements = visualize_with_measurements(img, outputs, save_path)

        if measurements:
            all_measurements.append(measurements)

        # Display
        cv2.imshow("Prediction with Measurements", result)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Summary statistics
    if all_measurements:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        avg_ratio = np.mean([m['ratio'] for m in all_measurements])
        avg_bottom_conf = np.mean([m['bottom_confidence'] for m in all_measurements])
        avg_top_conf = np.mean([m['top_confidence'] for m in all_measurements])

        print(f"Average bottom/top ratio: {avg_ratio:.2f}")
        print(f"Average top confidence: {avg_top_conf:.2f}")
        print(f"Average bottom confidence: {avg_bottom_conf:.2f}")

        if avg_ratio > 1.4:
            print("\n🔴 ISSUE DETECTED: Bottom baseline consistently too wide")
            print("   Recommended actions:")
            print("   1. Check annotation consistency")
            print("   2. Add more diverse training examples")
            print("   3. Apply geometric post-processing")


if __name__ == "__main__":
    main()
