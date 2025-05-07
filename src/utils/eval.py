import torch
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Union
import cv2

from src.utils.image_tools import get_image
from src.utils.metrics import calculate_pixel_iou_binary_classification
from src.utils.coco import COCODataset
from src.wsilib import WSITileContainer, WSITile, WSI, WSIDatabase

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
            "predictions": list[torch.Tensor],
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

    slide_dataset = COCODataset(
        annotation_file="/storage01/bolma/dev/data/datasets/WSI-ROI/slides/l8/annotations.json",
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

    # Create a mask from the tiles
    masks = {}
    for wsi_name, tiles in tqdm(tile_groups.items(), desc="Creating masks from tile groups"):
        # creating mask
        mask = torch.from_numpy(mask_from_tiles(tiles))
        # Convering masks to match the format of the two other methods
        masks[wsi_name] = crop_mask(mask, wsi_name, slide_dataset)

    results= {
        "inference_times": [],
        "ious": [],
        "predictions": [],
        "indices": []
    } 
    for key in tile_groups.keys():
        results["indices"].append(key)
        results["predictions"].append(masks[key])
        results["ious"].append(iou_results[key])
        results["inference_times"].append(inference_times[key])

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
    
    for wsi_name, tiles in wsi_groups.items(): 
        # Create a WSI object from the tiles
        wsi_path = Path("/storage01/bolma/dev/data/BIOMAG_slides/Lung") / (wsi_name[:wsi_name.rfind("HE")+2].upper().replace('-', '_') if wsi_name.rfind("HE") != -1 else wsi_name)
        wsi = WSI(wsi_path)

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
        
        iou_results[wsi_name] = iou
    
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

def mask_from_tiles(tiles: List[WSITile]) -> np.ndarray:
    """
    Create a mask from a list of WSITile objects.
    
    Args:
        tiles: List of WSITile objects
    
    Returns:
        Mask as a numpy array
    """
    
    level_height = 0
    level_width = 0
    for tile in tiles:
        if tile.y+tile.height > level_height:
            level_height = tile.y+tile.height
        if tile.x+tile.width > level_width:
            level_width = tile.x+tile.width
    
    mask = np.zeros((level_height, level_width), dtype=np.uint8)
    
    for tile in tiles:
        x, y, w, h = tile.x, tile.y, tile.width, tile.height
        mask[y:y+h, x:x+w] = tile.class_id
    
    return mask

def crop_mask(mask: torch.Tensor, wsi_name: str, dataset) -> torch.Tensor:
    """
    Crop the mask to match the dimensions of the masks of the other methods.

    Args:
        mask: Mask to be cropped (expected to be a binary tensor covering the full WSI area at source level)
        wsi_name: Name of the WSI
        dataset: Dataset containing the original images and their annotations
    Returns:
        Cropped mask aligned with the target image
    """
    # Get the crop offsets and target image dimensions
    crop_offsets = None
    target_level = None
    target_width = None
    target_height = None
    
    # Find the image metadata for this WSI
    for image_id in dataset.image_ids:
        img_info = dataset.coco.imgs[image_id]
        if img_info.get('wsi_source') == f"{wsi_name}.mrxs":
            crop_offsets = img_info.get('crop_offsets')
            target_level = img_info.get('level')
            target_width = img_info.get('width')
            target_height = img_info.get('height')
            break
    
    if crop_offsets is None or target_level is None:
        raise ValueError(f"Could not find crop offsets or level for WSI {wsi_name}")

    # Get source_level from the mask's original size
    # Assuming the mask is at level 6 (based on your eval.py code mentions)
    source_level = 6  # If this is different, adjust accordingly
    
    # Calculate the scaling factor between source and target level
    # Each level is typically 2x the resolution of the next level
    scale_factor = 2 ** (target_level - source_level)
    
    # Scale the mask to match target level's resolution
    if scale_factor != 1:
        # Resize the mask using interpolation
        # Use nearest neighbor to maintain binary values
        mask_height, mask_width = mask.shape
        scaled_width = int(mask_width / scale_factor)
        scaled_height = int(mask_height / scale_factor)
        
        # Reshape to [1, 1, H, W] for F.interpolate
        mask_reshaped = mask.unsqueeze(0).unsqueeze(0).float()
        
        # Use nearest interpolation to maintain binary mask
        scaled_mask = torch.nn.functional.interpolate(
            mask_reshaped,
            size=(scaled_height, scaled_width),
            mode='nearest'
        )
        
        # Back to original shape
        scaled_mask = scaled_mask.squeeze(0).squeeze(0)
    else:
        scaled_mask = mask
    
    # Now crop the scaled mask using the crop offsets
    x_start, y_start, x_end, y_end = crop_offsets
    
    # Make sure our scaled mask is large enough
    if scaled_mask.shape[0] < y_end or scaled_mask.shape[1] < x_end:
        # Pad if necessary
        pad_height = max(0, y_end - scaled_mask.shape[0])
        pad_width = max(0, x_end - scaled_mask.shape[1])
        scaled_mask = torch.nn.functional.pad(scaled_mask, (0, pad_width, 0, pad_height))
    
    # Crop the mask
    cropped_mask = scaled_mask[y_start:y_end, x_start:x_end]
    
    # Ensure the cropped mask matches the target dimensions
    if cropped_mask.shape[0] != target_height or cropped_mask.shape[1] != target_width:
        # Resize to match target dimensions exactly
        cropped_mask = torch.nn.functional.interpolate(
            cropped_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(target_height, target_width),
            mode='nearest'
        ).squeeze(0).squeeze(0)
    
    return cropped_mask

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
                # combine masks
                predicted_masks = prediction.get("masks")
                combined_mask = torch.zeros_like(predicted_masks[0])
                # Combine all masks
                for mask in predicted_masks:
                    combined_mask = torch.logical_or(combined_mask, mask)

                results["predictions"].append(combined_mask)

                # Metrics calculation
                #iou_best = calculate_pixel_iou_binary_classification(prediction, target, confidence_threshold,method="best")
                iou_combined = calculate_pixel_iou_binary_classification(prediction, target, confidence_threshold,method="combined")
                results["ious"].append(iou_combined)
                #metrics["iou_best"]+=iou_best
            
    return results