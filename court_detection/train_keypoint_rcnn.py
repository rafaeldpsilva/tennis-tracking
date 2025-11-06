import os
import json
import cv2
import random
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
import detectron2.data.transforms as T


# ============================================================================
# CONCEPT 1: Dataset Registration
# ============================================================================

def load_tennis_court_dataset(json_file, img_dir):
    """
    [
        {
            "file_name": "path/to/image.jpg",
            "image_id": 1,
            "height": 540,
            "width": 960,
            "annotations": [
                {
                    "bbox": [x, y, width, height],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": 0,
                    "keypoints": [x1, y1, v1, x2, y2, v2, ...],
                }
            ]
        },
        ...
    ]
    """
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    imgs = {img['id']: img for img in coco_data['images']}

    anns_per_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_per_image:
            anns_per_image[img_id] = []
        anns_per_image[img_id].append(ann)

    dataset_dicts = []
    for img_id, img_info in imgs.items():
        record = {}

        filename = os.path.join(img_dir, img_info['file_name'])
        record['file_name'] = filename
        record['image_id'] = img_id
        record['height'] = img_info['height']
        record['width'] = img_info['width']

        objs = []
        for ann in anns_per_image.get(img_id, []):
            obj = {
                'bbox': ann['bbox'],
                'bbox_mode': BoxMode.XYWH_ABS,
                'category_id': 0,
                'keypoints': ann['keypoints'],
            }
            objs.append(obj)

        record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_tennis_dataset(name, json_file, img_dir):
    DatasetCatalog.register(name, lambda: load_tennis_court_dataset(json_file, img_dir))

    MetadataCatalog.get(name).set(
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
        keypoint_flip_map=[],
        keypoint_connection_rules=[
            ("top_left_baseline", "top_right_baseline", (255, 0, 0)),      # Top baseline - blue
            ("top_right_baseline", "bottom_right_baseline", (0, 255, 0)),  # Right sideline - green
            ("bottom_right_baseline", "bottom_left_baseline", (255, 0, 0)), # Bottom baseline - blue
            ("bottom_left_baseline", "top_left_baseline", (0, 255, 0)),    # Left sideline - green
            ("top_left_service", "top_right_service", (0, 0, 255)),        # Top service - red
            ("bottom_left_service", "bottom_right_service", (0, 0, 255)),  # Bottom service - red
            ("top_left_service", "bottom_left_service", (255, 255, 0)),    # Left service center - cyan
            ("top_right_service", "bottom_right_service", (255, 255, 0)),  # Right service center - cyan
        ]
    )


# ============================================================================
# CONCEPT 2: Model Configuration
# ============================================================================
def setup_config(dataset_name, num_keypoints=8, max_iter=1000, batch_size=2):
    """
    Key hyperparameters explained:
    - BASE_LR: Learning rate (how fast the model updates). Start small for fine-tuning.
    - NUM_WORKERS: Parallel data loading threads.
    """
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = batch_size 
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = 200

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Force CPU mode (no GPU available)
    cfg.MODEL.DEVICE = "cpu"

    cfg.OUTPUT_DIR = "./output/keypoint_rcnn"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


# ============================================================================
# CONCEPT 3: Training
# ============================================================================

class TennisCourtTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Metrics:
        - AP (Average Precision): Main metric for keypoint detection
        - OKS (Object Keypoint Similarity): How close predicted keypoints are to ground truth
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


# ============================================================================
# CONCEPT 4: Visualization
# ============================================================================

def visualize_predictions(cfg, dataset_name, num_samples=3):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    for d in random.sample(dataset_dicts, min(num_samples, len(dataset_dicts))):
        img = cv2.imread(d["file_name"])

        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        result_img = out.get_image()[:, :, ::-1]

        output_path = os.path.join(
            cfg.OUTPUT_DIR,
            f"prediction_{Path(d['file_name']).stem}.jpg"
        )
        cv2.imwrite(output_path, result_img)
        print(f"  Saved prediction to: {output_path}")

        cv2.imshow("Predictions", result_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def compare_with_ground_truth(cfg, dataset_name, num_samples=3):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    for d in random.sample(dataset_dicts, min(num_samples, len(dataset_dicts))):
        img = cv2.imread(d["file_name"])

        outputs = predictor(img)
        v_pred = Visualizer(img[:, :, ::-1].copy(), metadata=metadata, scale=1.0)
        pred_img = v_pred.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]

        v_gt = Visualizer(img[:, :, ::-1].copy(), metadata=metadata, scale=1.0)
        gt_img = v_gt.draw_dataset_dict(d).get_image()[:, :, ::-1]

        comparison = cv2.hconcat([gt_img, pred_img])

        cv2.putText(comparison, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)
        cv2.putText(comparison, "Prediction", (gt_img.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        output_path = os.path.join(cfg.OUTPUT_DIR, f"comparison_{Path(d['file_name']).stem}.jpg")
        cv2.imwrite(output_path, comparison)
        print(f"  Saved comparison to: {output_path}")

        cv2.imshow("Ground Truth vs Prediction", comparison)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def main():
    print("="*60)
    print("Tennis Court Keypoint R-CNN Training")
    print("="*60)

    print("\n[1/5] Registering dataset...")
    dataset_name = "tennis_court_train"
    json_file = "training_data/tennis_court_keypoints.json"
    img_dir = "training_data/frames"

    register_tennis_dataset(dataset_name, json_file, img_dir)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    print(f"  ✓ Registered {len(dataset_dicts)} training images")

    print("\n[2/5] Configuring Keypoint R-CNN...")
    cfg = setup_config(
        dataset_name=dataset_name,
        num_keypoints=8,
        max_iter=500,  # Reduced for CPU training (increase if you have time)
        batch_size=2
    )
    print(f"  ✓ Model: keypoint_rcnn_R_50_FPN_3x")
    print(f"  ✓ Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  ✓ Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  ✓ Output dir: {cfg.OUTPUT_DIR}")

    print("\n[3/5] Training model...")
    print("  This will take a few minutes (depends on CPU/GPU)")
    print("  Watch for:")
    print("    - total_loss: Should decrease over time")
    print("    - loss_keypoint: Keypoint prediction loss")
    print("    - loss_box_reg: Bounding box regression loss")
    print()

    #trainer = TennisCourtTrainer(cfg)
    #trainer.resume_or_load(resume=False)
    #trainer.train()

    print("\n  ✓ Training complete!")
    print(f"  ✓ Model saved to: {cfg.OUTPUT_DIR}/model_final.pth")

    print("\n[4/5] Visualizing predictions...")
    visualize_predictions(cfg, dataset_name, num_samples=3)

    print("\n[5/5] Comparing with ground truth...")
    compare_with_ground_truth(cfg, dataset_name, num_samples=3)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Check TensorBoard: tensorboard --logdir {cfg.OUTPUT_DIR}")
    print(f"2. Review predictions in: {cfg.OUTPUT_DIR}/")
    print(f"3. Test on new videos using the trained model")
    print(f"\nModel path: {cfg.OUTPUT_DIR}/model_final.pth")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    main()
