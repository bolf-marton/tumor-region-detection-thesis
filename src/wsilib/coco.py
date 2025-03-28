import numpy as np
import os
import json
from skimage import measure


def convert_predictions_to_coco_format(predictions, annotations):
    """
    Converts Mask R-CNN predictions to COCO format.
    
    Args:
        predictions (list): List of predictions from the model.
        annotations (list): List of annotations for the images.
        
    Returns:
        coco_predictions (list): List of dictionaries in COCO format.
    """
    coco_predictions = []

    for i, prediction in enumerate(predictions):
        image_id = int(annotations[i]["image_id"].cpu().numpy()[0])
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        masks = prediction["masks"].cpu().numpy()

        for j in range(len(boxes)):
            bbox = boxes[j].tolist()
            score = float(scores[j])
            label = int(labels[j])
            mask = masks[j]

            # Convert bbox to COCO format [x, y, width, height]
            x_min, y_min, x_max, y_max = bbox
            bbox_coco = [x_min, y_min, x_max - x_min, y_max - y_min]

            """# Thresholding mask to binary
            binary_mask = (mask.squeeze(0) >= 0.5).astype(np.uint8)
            # Converting binary mask to RLE
            binary_mask_fortran = np.asfortranarray(binary_mask)
            rle = maskUtils.encode(binary_mask_fortran)
            # Converting bytes to string for JSON serialization
            rle['counts'] = rle['counts'].decode('utf-8')
            """

            contours = measure.find_contours(mask.squeeze(), 0.5)

            contour = np.flip(contours[0], axis=1)
            segmentation = contour.ravel().tolist()

            prediction_dict = {
                "image_id": image_id,
                "category_id": label,
                "bbox": bbox_coco,
                "score": score,
                "segmentation": [segmentation]
            }

            coco_predictions.append(prediction_dict)

    return coco_predictions

def save_predictions_in_coco_format(predictions, annotations, save_path):
    coco_predictions = convert_predictions_to_coco_format(predictions, annotations)
    with open(os.path.join(save_path, 'predictions.json'), 'w') as f:
        json.dump(coco_predictions, f)