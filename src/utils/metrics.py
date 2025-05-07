import torch
from shapely.geometry import Polygon

def calculate_pixel_iou_binary_classification(predictions, targets, confidence_threshold=0.5, mask_threshold=0.5, method:str="best"):
    """
    Calculate the pixelvise Intersection over Union (IoU). Firstly, the predicted and target masks are merged and
    these two masks are then compared. This is only applicable if there are only two classes: background and tumor.

    Args:
        predictions (dict): A dictionary containing the predicted binary masks, in the form of a list of PyTorch tensors 
                            (each of shape (H, W)).
        targets (dict): A dictionary containing the ground truth binary masks, in the form of a list of PyTorch tensors 
                        (each of shape (H, W)).
        confidence_threshold (float): The predictions having lower scores are ommitted from the IoU calculation. Defaults to 0.5.
        mask_threshold (float): Threshold to binarize the masks. Defaults to 0.5.
        method (str): The method to use for IoU calculation. Can be "best" or "combined".
            - "best": Use the best mask from predictions and targets.
            - "combined": Combine all masks from predictions and targets.

    Returns:
        float: The IoU score.
    """
    predicted_masks = predictions.get("masks")
    target_masks = targets.get("masks")

    # Handling empty target or prediction
    if len(predicted_masks) == 0 or len(target_masks) == 0:
        return 0.0  # Returning 0 IoU if either prediction or target is empty

    if "scores" in predictions.keys():
        if method == "best":
            # Use the best mask from predictions
            scores = predictions.get("scores").tolist()
            best_score = max(scores)
            best_index = scores.index(best_score)
            pred_mask = predicted_masks[best_index]
            # Use the best mask from targets
            target_mask = target_masks[best_index]

        elif method == "combined":
            # Use one combined mask from all predictions
            pred_mask = torch.zeros_like(predicted_masks[0])
            
            # Combine masks using maximum value at each pixel
            for mask in predicted_masks:
                pred_mask = torch.max(pred_mask, mask)
            
            # Threshold the COMBINED mask
            pred_mask = (pred_mask > confidence_threshold).to(torch.uint8)

            target_mask = torch.zeros_like(target_masks[0])

            # Combine masks using maximum value at each pixel
            for mask in target_masks:
                target_mask = torch.max(target_mask, mask)
            
            # Threshold the COMBINED target mask
            target_mask = (target_mask > confidence_threshold).to(torch.uint8)
    else:
        # Use the first mask from predictions and targets
        pred_mask = predicted_masks[0]
        target_mask = target_masks[0]
        

    # Calculate intersection and union
    intersection = torch.logical_and(pred_mask, target_mask).sum().item()
    union = torch.logical_or(pred_mask, target_mask).sum().item()

    # Calculate IoU
    iou = intersection / union if union != 0 else 0.0

    return iou