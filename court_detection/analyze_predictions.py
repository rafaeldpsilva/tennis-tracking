"""
Analyze keypoint prediction errors to understand model weaknesses.

This script helps you understand WHERE and WHY the model makes mistakes.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_coco_annotations(json_file):
    """Load ground truth annotations."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def calculate_keypoint_errors(predictions, ground_truth):
    """
    Calculate pixel-wise error for each keypoint.

    Returns:
        errors: Dict mapping keypoint name to list of errors
        stats: Summary statistics
    """
    keypoint_names = [
        "top_left_baseline",
        "top_right_baseline",
        "bottom_left_baseline",
        "bottom_right_baseline",
        "top_left_service",
        "top_right_service",
        "bottom_left_service",
        "bottom_right_service",
    ]

    errors = {name: [] for name in keypoint_names}

    # Calculate Euclidean distance for each keypoint
    for i, name in enumerate(keypoint_names):
        gt_x, gt_y, gt_v = ground_truth[i*3:i*3+3]
        pred_x, pred_y, pred_v = predictions[i*3:i*3+3]

        if gt_v > 0 and pred_v > 0:  # Both visible
            error = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
            errors[name].append(error)

    return errors


def analyze_baseline_width_bias(predictions, ground_truth):
    """
    Specifically analyze if bottom baseline is predicted wider than top.

    Concept: In tennis court perspective, bottom (near camera) should be
    wider than top (far from camera), but model might exaggerate this.
    """
    # Top baseline width (ground truth)
    top_left_gt = ground_truth[0:2]
    top_right_gt = ground_truth[3:5]
    gt_top_width = np.linalg.norm(np.array(top_right_gt) - np.array(top_left_gt))

    # Bottom baseline width (ground truth)
    bottom_left_gt = ground_truth[6:8]
    bottom_right_gt = ground_truth[9:11]
    gt_bottom_width = np.linalg.norm(np.array(bottom_right_gt) - np.array(bottom_left_gt))

    # Predicted widths
    top_left_pred = predictions[0:2]
    top_right_pred = predictions[3:5]
    pred_top_width = np.linalg.norm(np.array(top_right_pred) - np.array(top_left_pred))

    bottom_left_pred = predictions[6:8]
    bottom_right_pred = predictions[9:11]
    pred_bottom_width = np.linalg.norm(np.array(bottom_right_pred) - np.array(bottom_left_pred))

    # Calculate ratios
    gt_ratio = gt_bottom_width / gt_top_width if gt_top_width > 0 else 0
    pred_ratio = pred_bottom_width / pred_top_width if pred_top_width > 0 else 0

    results = {
        'gt_top_width': gt_top_width,
        'gt_bottom_width': gt_bottom_width,
        'gt_ratio': gt_ratio,
        'pred_top_width': pred_top_width,
        'pred_bottom_width': pred_bottom_width,
        'pred_ratio': pred_ratio,
        'width_bias': pred_bottom_width - gt_bottom_width,
        'ratio_error': pred_ratio - gt_ratio,
    }

    return results


def visualize_error_distribution(all_errors):
    """
    Create visualization showing which keypoints have highest errors.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Box plot of errors per keypoint
    keypoint_names = list(all_errors.keys())
    error_data = [all_errors[name] for name in keypoint_names]

    ax1.boxplot(error_data, labels=keypoint_names)
    ax1.set_ylabel('Pixel Error')
    ax1.set_title('Keypoint Prediction Error Distribution')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Mean error per keypoint
    mean_errors = [np.mean(errors) if errors else 0 for errors in error_data]
    ax2.bar(keypoint_names, mean_errors, color='skyblue', edgecolor='navy')
    ax2.set_ylabel('Mean Pixel Error')
    ax2.set_title('Average Error Per Keypoint')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Highlight baseline keypoints
    baseline_indices = [0, 1, 2, 3]  # top/bottom left/right baseline
    for idx in baseline_indices:
        ax2.get_children()[idx].set_color('coral')

    plt.tight_layout()
    plt.savefig('output/keypoint_error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved error visualization to: output/keypoint_error_analysis.png")


def print_diagnosis(width_analyses):
    """
    Print human-readable diagnosis of the baseline width issue.
    """
    print("\n" + "="*60)
    print("BASELINE WIDTH ANALYSIS")
    print("="*60)

    avg_width_bias = np.mean([w['width_bias'] for w in width_analyses])
    avg_ratio_error = np.mean([w['ratio_error'] for w in width_analyses])

    print(f"\nAverage bottom baseline width bias: {avg_width_bias:.1f} pixels")
    print(f"Average ratio error: {avg_ratio_error:.3f}")

    if avg_width_bias > 20:
        print("\n⚠️  DIAGNOSIS: Significant width bias detected!")
        print("   The model is predicting bottom baseline too wide.")
        print("\n🔧 RECOMMENDED FIXES:")
        print("   1. Annotate more frames with varied camera angles")
        print("   2. Add data augmentation (perspective transforms)")
        print("   3. Post-process with geometric constraints")
    elif avg_width_bias > 10:
        print("\n⚠️  DIAGNOSIS: Moderate width bias")
        print("   Model slightly overestimates bottom baseline width")
        print("\n🔧 RECOMMENDED FIXES:")
        print("   1. Add 5-10 more training examples")
        print("   2. Verify annotation consistency")
    else:
        print("\n✅ DIAGNOSIS: Width predictions are reasonable!")
        print("   Bottom baseline bias is within acceptable range")

    print("\n" + "="*60)


def main():
    """
    Analyze model predictions vs ground truth.

    This assumes you have:
    - Ground truth: training_data/tennis_court_keypoints.json
    - Predictions: From running inference with trained model
    """
    # For now, load ground truth to show you the analysis structure
    # You'll need to add actual predictions from your trained model

    print("Keypoint Prediction Error Analysis")
    print("="*60)

    gt_file = "training_data/tennis_court_keypoints.json"
    coco_data = load_coco_annotations(gt_file)

    print(f"\nLoaded {len(coco_data['annotations'])} annotations")

    # Simulate analysis (you'll replace this with actual predictions)
    print("\nTo complete this analysis, you need to:")
    print("1. Run inference on training images with your trained model")
    print("2. Save predictions")
    print("3. Compare with ground truth")

    print("\nExample code to get predictions:")
    print("""
    from detectron2.engine import DefaultPredictor

    predictor = DefaultPredictor(cfg)
    predictions = []

    for annotation in coco_data['annotations']:
        img_path = ...  # Get image path
        img = cv2.imread(img_path)
        outputs = predictor(img)
        predictions.append(outputs["instances"].pred_keypoints[0])
    """)


if __name__ == "__main__":
    main()
