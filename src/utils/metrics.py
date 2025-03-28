import torch

def calculate_pixel_iou_binary_classification(predictions, targets, confidence_threshold=0.5, mask_threshold=0.5):
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

    Returns:
        float: The IoU score.
    """
    predicted_masks = predictions.get("masks",[])
    prediction_scores = predictions.get("scores",[])
    target_masks = targets.get("masks",[])

    # Handling empty target or predicion
    if len(predicted_masks) == 0 or len(target_masks) == 0:
        return 0.0  # Returning 0 IoU if either prediction or target is empty

    # Initialize merged masks with zeros
    merged_pred_mask = torch.zeros_like(predicted_masks[0], dtype=torch.uint8)
    merged_target_mask = torch.zeros_like(target_masks[0], dtype=torch.uint8)

    # Merge all predicted masks
    for pred_mask, score in zip(predicted_masks, prediction_scores):
        if score >= confidence_threshold:
            merged_pred_mask = torch.logical_or(merged_pred_mask, (pred_mask > mask_threshold).to(torch.uint8))

    # Merge all target masks
    for target_mask in target_masks:
        merged_target_mask = torch.logical_or(merged_target_mask, (target_mask > mask_threshold).to(torch.uint8))

    # Calculate intersection and union
    intersection = torch.logical_and(merged_pred_mask, merged_target_mask).sum().item()
    union = torch.logical_or(merged_pred_mask, merged_target_mask).sum().item()

    # Calculate IoU
    iou = intersection / union if union != 0 else 0.0

    return iou