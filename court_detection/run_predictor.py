from detectron2.engine import DefaultPredictor

predictor = DefaultPredictor(cfg)
predictions = []

for annotation in coco_data['annotations']:
    img_path = ...  # Get image path
    img = cv2.imread(img_path)
    outputs = predictor(img)
    predictions.append(outputs["instances"].pred_keypoints[0])