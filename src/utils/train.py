import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from tqdm import tqdm
import os
from torchvision.ops import nms

import numpy as np

from assets.dataset import get_image

from fastai.vision.data import DataLoaders

from assets.evaluation import evaluate_retinanet, calculate_pixel_iou_binary_classification, calculate_pixel_dice_binary_classification
from assets.coco import save_predictions_in_coco_format


def train_maskrcnn(model, device:str, dataloaders:DataLoaders, optimizer):
    """Training script for Mask R-CNN.

    Args:
        model: pytorch NN model.
        device: "cuda" / "cpu".
        dataloaders: fastai DataLoaders.
        optimizer: the desired optimizer.

    Returns:
        model, 
        metrics: {"training_loss":mean_loss}
        
    """

    # Sending model to device
    model = model.to(device)

    model.train()  # Set model to training mode
    
    dataloader = dataloaders.train

    loss = 0.0
    mean_loss = 0.0

    # Iterate over the data
    for image, targets in tqdm(dataloader):
        
        # preprocessing and converting to tensors
        image = [torch.tensor(get_image(image[0])).squeeze().permute(2,0,1).to(device)]
        targets = [{key: targets[0][key].squeeze(0).to(device) for key in targets[0].keys()}]

        # Zero the variables that are used for storing the gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward
        with torch.set_grad_enabled(True):

            loss_dict = model(image, targets)
            """
            {
            'loss_classifier': tensor(0.7220, device='cuda:0', grad_fn=<NllLossBackward0>), 
            'loss_box_reg': tensor(0.1060, device='cuda:0', grad_fn=<DivBackward0>), 
            'loss_mask': tensor(0.6930, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 
            'loss_objectness': tensor(0.0359, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 
            'loss_rpn_box_reg': tensor(0.0006, device='cuda:0', grad_fn=<DivBackward0>)}
            }
            """
            loss = sum(loss for loss in loss_dict.values())

            mean_loss += loss

            # Backward + optimize (only if in training phase)
            loss.backward()
            optimizer.step()
    

    mean_loss = mean_loss / len(dataloader.dataset)
    
    print(f"Train – Mean Loss: {mean_loss:.4f}") # , IoU: {IoU:.4f}

    metrics = {"training_loss":mean_loss}
                
    return model, metrics

def test_maskrcnn(model, device:str, dataloaders:DataLoaders, optimizer, confidence_threshold: float=0.5):
    """Runs testing for the model.

    Args:
        model: pytorch NN model.
        device: "cuda" / "cpu".
        dataloaders: fastai DataLoaders.
        optimizer: the desired optimizer.
        confidence_threshold: confidence score threshold to filter predictions during evaluation.

    Returns:
        metrics: { 
                "test_loss" : float,
                "iou": float,
                "dice": float
                }
    """

    # Sending model to device
    model = model.to(device)

    # for validation
    #mAP = MeanAveragePrecision(box_format='xyxy',iou_type='bbox')

    dataloader = dataloaders.valid

    metrics = { 
        "test_loss" : 0.0,
        "iou": 0.0,
        "dice" : 0.0
        }

    loss = 0.0

    # Iterate over the data
    for image, targets in tqdm(dataloader):

        # preprocessing and converting to tensors
        image = [torch.tensor(get_image(image[0])).squeeze().permute(2,0,1).to(device)]
        targets = [{key: targets[0][key].squeeze(0).to(device) for key in targets[0].keys()}]
        
        # Zero the variables that are used for storing the gradients
        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(False):
            # To get training loss
            model.train()  # Set model to training mode

            loss_dict = model(image, targets)
            """{
            'loss_classifier': tensor(0.7220, device='cuda:0', grad_fn=<NllLossBackward0>), 
            'loss_box_reg': tensor(0.1060, device='cuda:0', grad_fn=<DivBackward0>), 
            'loss_mask': tensor(0.6930, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 
            'loss_objectness': tensor(0.0359, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 
            'loss_rpn_box_reg': tensor(0.0006, device='cuda:0', grad_fn=<DivBackward0>)}
            }"""
            loss = sum(loss for loss in loss_dict.values())

            metrics["test_loss"]+=loss

            # For other metrics
            model.eval()  # Set model to evaluate mode
            predictions = model(image)

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
                # Metrics calculation
                iou = calculate_pixel_iou_binary_classification(prediction, target, confidence_threshold)
                dice = calculate_pixel_dice_binary_classification(prediction, target, confidence_threshold)
                metrics["iou"]+=iou
                metrics["dice"]+=dice

    
    for k in metrics.keys():
        metrics[k] = metrics[k] / len(dataloader.dataset)

    print(f"Test – IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")
                    
    return metrics