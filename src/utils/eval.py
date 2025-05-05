import torch
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Union

from src.utils.image_tools import get_image
from src.utils.metrics import calculate_pixel_iou_binary_classification
from src.utils.coco import COCODataset
from src.wsilib import WSITileContainer, WSITile, WSI

# RESNET

def resnet_eval(model: Any, device: str, eval_dataset_annotation_path: Path) -> Union[Dict[str, Dict], Dict[str, List[WSITile]]]:
    """Runs evaluation for ResNet classification model based on the IoU of the predicted tiles.
    Each tile treated as each pixel of the image is belonging to the predicted class of the tile.
    The IoU is calculated between this mask and the ground truth mask.

    Args:
        model: ResNet model
        device: "cuda" / "cpu"
        eval_dataset_annotation_path: Path to the evaluation dataset annotation file.

    Returns:
        results: { 
            "inference_times": list[float],
            "ious": list[float],
            "predictions": list[list[WSITile]],
            "indices": list[str]
        }
    """
    # Load the tiles dataset
    dataset = COCODataset(
        annotation_file=eval_dataset_annotation_path,
        train=False,
        split_file=Path("/storage01/bolma/dev/tumor-region-detection-thesis/dataset_split.json"),
        random_seed=42,
        transform=None,
    )

    # Create WSI tiles from COCO dataset with predictions
    tile_groups = create_wsi_tiles_from_coco_with_prediction(dataset, model, device)
    # Calculate IoU for each WSI

    iou_results = calculate_iou_for_tiles(tile_groups)

    # Extract average inference times from tile groups
    inference_times = get_inference_times(tile_groups)

    metrics = {
        "inference_times": inference_times,
        "ious": iou_results
    }

    results = {
        "inference_times": [inference_time for inference_time in inference_times.values()],
        "ious": [iou for iou in iou_results.values()],
        "predictions": [tiles for tiles in tile_groups.values()],
        "indices": [name for name in tile_groups.keys()]
    }

    return results

def create_wsi_tiles_from_coco_with_prediction(
    coco_dataset: COCODataset, 
    model: Any,
    device: str,
) -> Dict[str, List[WSITile]]:
    """
    Create lists of WSITile objects from a COCO dataset, grouped by source WSIs.
    
    Args:
        coco_dataset: COCODataset object containing the annotations
        model: Model to use for prediction
        device: Device to run the model on (e.g., "cuda" or "cpu")
        
    Returns:
        Dict with source WSI names an list of WSITile objects
    """
    wsi_groups = {}

    # prepare model for evaluation
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Iterate through all images in the COCO dataset
    for image_id in tqdm(coco_dataset.image_ids, desc="Processing images"):
        # Get the image info from COCO
        coco_img = coco_dataset.coco.imgs[image_id]
        
        # Extract WSI source
        wsi_source = coco_img.get('wsi_source', None)
            
        # Add to the appropriate group
        if wsi_source not in wsi_groups.keys():
            wsi_groups[wsi_source] = []

        # Load annotation for the tile (one class for each tile)
        ann_id = coco_dataset.coco.getAnnIds(imgIds=image_id)
        if len(ann_id) == 0:
            gt_category_id = 0
        else:
            annotation = coco_dataset.coco.loadAnns(ann_id)
            gt_category_id = annotation[0]['category_id']

        # PREDICTION
        # get image for prediciton
        image_path = coco_dataset.annotation_file.parent / coco_dataset.coco.imgs[image_id]['file_name']
        image = Image.open(image_path).convert('RGB')

        image_tensor = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            start_time = time.time()
            output = model(image_tensor)
            end_time = time.time()
            _, predicted_class = torch.max(output, 1)
            predicted_class = predicted_class.item()
        
        # Append WSITile object to its group
        wsi_groups[wsi_source].append(WSITile(coco_img['level'], 
                                              x=coco_img['tile_x'], 
                                              y=coco_img['tile_y'], 
                                              width=coco_img['width'], 
                                              height=coco_img['height'], 
                                              image_path=Path(image_path), 
                                              class_id=gt_category_id, 
                                              metadata={"id": image_id, "predicted_class":predicted_class, "inference_time": end_time - start_time}))

    print(f"Found {len(wsi_groups)} unique WSI sources.")
        
    return wsi_groups

def calculate_iou_for_tiles(
    wsi_groups: Dict[str, List[WSITile]],
) -> Dict[str, float]:
    """
    Calculate IoU for each WSI group.
    
    Args:
        wsi_groups: Dictionary with WSI source names and list of WSITile objects
    
    Returns:
        Dictionary with WSI source names and their corresponding IoU values
    """
    iou_results = {}
    
    for wsi_source, tiles in wsi_groups.items():
        # Create a WSI object from the tiles
        wsi_source = Path("/storage01/bolma/dev/data/BIOMAG_slides/Lung") / (wsi_source[:wsi_source.rfind("HE")+2].upper().replace('-', '_') if wsi_source.rfind("HE") != -1 else wsi_source)
        wsi = WSI(wsi_source)

        level_width = wsi.level_dimensions[tiles[0].level][0]
        level_height = wsi.level_dimensions[tiles[0].level][1]
        
        # Calculate IoU
        gt_mask = np.zeros((level_height, level_width), dtype=np.uint8)
        pred_mask = np.zeros((level_height, level_width), dtype=np.uint8)
        
        # Add ground truth to mask
        for tile in tiles:
            x, y, w, h = tile.x, tile.y, tile.width, tile.height
            gt_mask[y:y+h, x:x+w] = tile.class_id
        
        # Add prediction to mask
        for tile in tiles:
            x, y, w, h = tile.x, tile.y, tile.width, tile.height
            pred_mask[y:y+h, x:x+w] = tile.metadata["predicted_class"]

        gt_bool = gt_mask > 0
        pred_bool = pred_mask > 0

        intersection = np.logical_and(gt_bool, pred_bool).sum()
        union = np.logical_or(gt_bool, pred_bool).sum() 
        iou = intersection / union if union > 0 else 0.0
        
        iou_results[wsi_source] = iou
    
    return iou_results

def get_inference_times(tile_groups: Dict[str, List[WSITile]]) -> Dict[str, float]:
    """
    Calculate the mean inference times for each tile group.
    
    Args:
        tile_groups: Dictionary with WSI source names and list of WSITile objects
    
    Returns:
        Dict[wsi_name: mean inference time in seconds]
    """

    inference_times = {}
    for wsi_name, tiles in tile_groups.items():
        total_inference_time = 0.0
        num_tiles = 0
        
        for tile in tiles:
            total_inference_time += tile.metadata["inference_time"]
            num_tiles += 1
            
        mean_inference_time = (total_inference_time / num_tiles) * 1000 if num_tiles > 0 else 0.0
        inference_times[wsi_name] = mean_inference_time

    return inference_times

# UNET

def unet_eval(model: Any, device: str, dataloader: torch.utils.data.DataLoader, confidence_threshold: float = 0.5) -> Dict[str, List]:
    """Runs evaluation for UNet model.

    Args:
        model: UNet model
        device: "cuda" / "cpu"
        dataloader: DataLoader for testing
        confidence_threshold: Threshold for binary mask prediction

    Returns:
        results: { 
                "inference_times": list[float],
                "ious": list[float]
                "predictions": list[torch.Tensor]
                "indices": list[int]
                }
    """
    # Sending model to device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    results = {
        "inference_times": [],
        "ious": [],
        "predictions": [],
        "indices": []
    }

    # Iterate over the data
    for idx, (images, targets) in tqdm(enumerate(dataloader)):
        results["indices"].append(idx)
        # Preprocessing and converting to tensors
        image = images.to(device)
        target_masks = targets['masks'].squeeze(0)

        binary_mask = torch.zeros(
            (image.shape[0], 1, target_masks.shape[1], target_masks.shape[2]),  # Shape: [B, 1, H, W]
            device=device
        )
        
        # Combine instance masks into one binary mask
        if target_masks.shape[0] > 0:  # If we have any masks
            binary_mask[0] = (target_masks.sum(dim=0) > 0).float()

        with torch.no_grad():

            start_time = time.time()
            output = model(image)
            end_time = time.time()

            # Calculate inference time
            results["inference_times"].append(end_time - start_time)
            
            pred_mask = (output > confidence_threshold).float()

            results["predictions"].append(pred_mask)
            
            # Calculate IoU (intersection over union)
            intersection = torch.logical_and(pred_mask, binary_mask).sum().float()
            union = torch.logical_or(pred_mask, binary_mask).sum().float()
            
            # Handle edge case of empty masks
            if union > 0:
                iou = intersection / union
            else:
                iou = torch.tensor(1.0 if binary_mask.sum() == 0 else 0.0, device=device)
                
            results["ious"].append(iou.item())
    
    return results

# MASK-RCNN
def maskrcnn_eval(model:Any, device:str, dataloader:torch.utils.data.DataLoader, confidence_threshold: float=0.5) -> Dict[str, List]:
    """Runs evaluation for the MaskRCNN model.

    Args:
        model (Any): pytorch NN model.
        device (str): "cuda" / "cpu".
        dataloader (torch.utils.data.DataLoader): DataLoader for testing.
        confidence_threshold (float): confidence score threshold to filter predictions during evaluation.

    Returns:
        results: { 
            "inference_times": list[float],
            "ious": list[float]
            "predictions": list[torch.Tensor]
            "indices": list[int]
            }
    """

    # Sending model to device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    results = {
        "inference_times": [],
        "ious": [],
        "predictions": [],
        "indices": []
    }

    # Iterate over the data
    for idx, (images, targets) in tqdm(enumerate(dataloader)):
        results["indices"].append(idx)
        # preprocessing and converting to tensors
        images = [images[0].to(device)]
        targets = [{key: targets[key].squeeze(0).to(device) for key in targets.keys()}]

        with torch.no_grad():
            
            start_time = time.time()
            predictions = model(images)
            end_time = time.time()

            # Calculate inference time
            results["inference_times"].append(end_time - start_time)

            """
            PREDICTION DICT:
            - "boxes" (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
            ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            - "labels" (Int64Tensor[N]): the predicted labels for each image
            - "scores" (Tensor[N]): the scores or each prediction
            - "masks" (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to
            obtain the final segmentation masks, the soft masks can be thresholded, generally
            with a value of 0.5 (mask >= 0.5)
            """
            
            for prediction, target in zip(predictions, targets):
                results["predictions"].append(prediction)
                # Metrics calculation
                #iou_best = calculate_pixel_iou_binary_classification(prediction, target, confidence_threshold,method="best")
                iou_combined = calculate_pixel_iou_binary_classification(prediction, target, confidence_threshold,method="combined")
                results["ious"].append(iou_combined)
                #metrics["iou_best"]+=iou_best
            
    return results