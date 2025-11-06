"""
Convert LabelMe keypoint annotations to COCO format for Detectron2 training.

COCO Keypoint Format Explanation:
- Each annotation has a 'keypoints' field: [x1, y1, v1, x2, y2, v2, ...]
- v (visibility): 0=not labeled, 1=labeled but occluded, 2=labeled and visible
- 'bbox': [x, y, width, height] bounding box around all keypoints
- 'num_keypoints': count of visible keypoints (v > 0)
"""

import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime


# Define the expected keypoints in order (14 total for complete court)
KEYPOINT_NAMES = [
    # Baseline corners (at doubles sidelines)
    "top_left_doubles",           # 0
    "top_right_doubles",          # 1
    "bottom_left_doubles",        # 2
    "bottom_right_doubles",       # 3

    # Baseline × Singles sidelines
    "top_left_singles",           # 4
    "top_right_singles",          # 5
    "bottom_left_singles",        # 6
    "bottom_right_singles",       # 7

    # Service lines × Singles sidelines
    "top_left_service",           # 8
    "top_right_service",          # 9
    "bottom_left_service",        # 10
    "bottom_right_service",       # 11

    # Center service line intersections
    "top_center_service",         # 12
    "bottom_center_service",      # 13
]

# Define skeleton connections (for visualization)
# Each pair shows which keypoints are connected by lines
SKELETON = [
    [0, 1],  # top baseline
    [1, 3],  # right sideline
    [3, 2],  # bottom baseline
    [2, 0],  # left sideline
    [4, 5],  # top service line
    [6, 7],  # bottom service line
    [4, 6],  # left service center line
    [5, 7],  # right service center line (not typically visible, but logical)
]


def load_labelme_json(json_path):
    """Load a LabelMe JSON annotation file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def labelme_to_coco_annotation(labelme_data, image_id, annotation_id):
    """
    Convert a single LabelMe annotation to COCO format.

    Args:
        labelme_data: Parsed LabelMe JSON data
        image_id: Unique ID for this image
        annotation_id: Unique ID for this annotation

    Returns:
        Dict in COCO annotation format, or None if invalid
    """
    # Extract image dimensions
    image_height = labelme_data['imageHeight']
    image_width = labelme_data['imageWidth']

    # Create a dictionary to store keypoints by name
    keypoints_dict = {}

    for shape in labelme_data['shapes']:
        if shape['shape_type'] == 'point':
            label = shape['label']
            x, y = shape['points'][0]  # Point coordinates
            keypoints_dict[label] = (x, y, 2)  # visibility=2 (visible)

    # Build the keypoints array in the correct order
    keypoints = []
    num_keypoints = 0

    for kp_name in KEYPOINT_NAMES:
        if kp_name in keypoints_dict:
            x, y, v = keypoints_dict[kp_name]
            keypoints.extend([x, y, v])
            num_keypoints += 1
        else:
            # Keypoint not annotated
            keypoints.extend([0, 0, 0])

    # Calculate bounding box around visible keypoints
    visible_points = [
        (keypoints[i], keypoints[i+1])
        for i in range(0, len(keypoints), 3)
        if keypoints[i+2] > 0  # visibility > 0
    ]

    if not visible_points:
        print(f"  Warning: No visible keypoints found in {labelme_data['imagePath']}")
        return None

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add padding to bbox
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width, x_max + padding)
    y_max = min(image_height, y_max + padding)

    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    # Calculate area (required by COCO format)
    area = (x_max - x_min) * (y_max - y_min)

    # Create COCO annotation
    coco_annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,  # We only have one category: tennis_court
        "keypoints": keypoints,
        "num_keypoints": num_keypoints,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
    }

    return coco_annotation


def convert_labelme_to_coco(labelme_dir, output_json, image_dir=None):
    """
    Convert all LabelMe annotations in a directory to a COCO dataset JSON.

    Args:
        labelme_dir: Directory containing LabelMe JSON files
        output_json: Path to output COCO JSON file
        image_dir: Directory containing images (if different from labelme_dir)
    """
    if image_dir is None:
        image_dir = labelme_dir

    labelme_dir = Path(labelme_dir)
    image_dir = Path(image_dir)

    # Find all LabelMe JSON files
    json_files = list(labelme_dir.glob("*.json"))

    print(f"Found {len(json_files)} annotation files")

    # Initialize COCO dataset structure
    coco_dataset = {
        "info": {
            "description": "Tennis Court Keypoint Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "tennis_court",
                "keypoints": KEYPOINT_NAMES,
                "skeleton": SKELETON,
            }
        ],
    }

    annotation_id = 1

    for idx, json_file in enumerate(json_files, start=1):
        labelme_data = load_labelme_json(json_file)

        # Add image info
        image_filename = labelme_data['imagePath']
        image_info = {
            "id": idx,
            "file_name": image_filename,
            "height": labelme_data['imageHeight'],
            "width": labelme_data['imageWidth'],
        }
        coco_dataset["images"].append(image_info)

        # Convert annotation
        coco_annotation = labelme_to_coco_annotation(labelme_data, idx, annotation_id)
        visualize_annotation(Path(image_dir) / image_filename, output_json, idx)
        if coco_annotation is not None:
            coco_dataset["annotations"].append(coco_annotation)
            annotation_id += 1
            print(f"  ✓ Converted {image_filename} ({coco_annotation['num_keypoints']} keypoints)")
        else:
            print(f"  ✗ Skipped {image_filename} (invalid)")

    # Save COCO JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COCO dataset saved to: {output_json}")
    print(f"Total images: {len(coco_dataset['images'])}")
    print(f"Total annotations: {len(coco_dataset['annotations'])}")
    print(f"\nDataset ready for Detectron2 training!")


def visualize_annotation(image_path, coco_json, image_id):
    """
    Visualize COCO keypoint annotations on an image.

    Args:
        image_path: Path to the image
        coco_json: Path to COCO annotations JSON
        image_id: ID of the image to visualize
    """
    import cv2

    # Load COCO data
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # Find the annotation for this image
    annotation = None
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            annotation = ann
            break

    if annotation is None:
        print(f"No annotation found for image_id {image_id}")
        return

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    # Draw keypoints
    keypoints = annotation['keypoints']
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v > 0:  # Visible
            kp_idx = i // 3
            color = (0, 255, 0) if v == 2 else (0, 165, 255)  # Green if visible, orange if occluded
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
            # Draw keypoint label
            cv2.putText(image, str(kp_idx), (int(x)+10, int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw skeleton
    category = coco_data['categories'][0]
    skeleton = category['skeleton']
    for connection in skeleton:
        kp1_idx, kp2_idx = connection
        kp1_i = kp1_idx * 3
        kp2_i = kp2_idx * 3

        if keypoints[kp1_i+2] > 0 and keypoints[kp2_i+2] > 0:
            pt1 = (int(keypoints[kp1_i]), int(keypoints[kp1_i+1]))
            pt2 = (int(keypoints[kp2_i]), int(keypoints[kp2_i+1]))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)

    # Draw bounding box
    bbox = annotation['bbox']
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Display
    cv2.imshow("Keypoint Annotation Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Convert annotations
    convert_labelme_to_coco(
        labelme_dir="training_data/frames",
        output_json="training_data/tennis_court_keypoints.json"
    )

    print("\nTo visualize annotations, you can use:")
    print("  python convert_to_coco.py --visualize")
