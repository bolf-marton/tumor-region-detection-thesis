import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Any

from src.utils.comet import CometLogger
from src.utils.image_tools import get_image
from src.utils.metrics import calculate_pixel_iou_binary_classification

def maskrcnn_training_testing_loop(model:Any, device:str, train_dataloader:torch.utils.data.DataLoader, optimizer:Any, scheduler:Any, n_epochs:int=0, box_score_thresh:float=0.5, comet_logger:CometLogger=None):
    """Training loop for Mask R-CNN.
    Args:
        model (Any): pytorch NN model.
        device (str): "cuda" / "cpu".
        train_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for training.
        optimizer (Any): the desired optimizer.
        scheduler (Any): the desired scheduler.
        n_epochs (int): number of epochs to train.
        box_score_thresh (float): confidence score threshold for bounding box predictions.
        comet_logger (CometLogger): Comet.ml logger (optional).
    Returns:
        model: trained model.
        metrics: { 
                "test_loss" : float,
                "iou": float,
                "dice": float
                }
    """
    if n_epochs != 0:
            print("\n🚀 STARTING TRAINING")

            for epoch in range(n_epochs):
                if comet_logger is not None:
                    print("-" * len(f"Epoch {epoch+1}/{n_epochs} - {comet_logger.name}"))
                    print(f"Epoch {epoch+1}/{n_epochs} - {comet_logger.name}")
                    print("-" * len(f"Epoch {epoch+1}/{n_epochs} - {comet_logger.name}"))
                else:
                    print("-" * len(f"Epoch {epoch+1}/{n_epochs}"))
                    print(f"Epoch {epoch+1}/{n_epochs}")
                    print("-" * len(f"Epoch {epoch+1}/{n_epochs}"))

                # TRAINING
                model, train_metrics = train_maskrcnn(model,device,train_dataloader,optimizer)
                
                # Log metrics to Comet.ml
                if comet_logger is not None:
                    for k in train_metrics.keys():
                        comet_logger.log_metric(k, train_metrics[k], epoch=epoch + 1)


                # TESTING
                test_metrics = test_maskrcnn(model,device,train_dataloader,optimizer,box_score_thresh)

                # Log metrics to Comet.ml
                if comet_logger is not None:
                    for k in test_metrics.keys():
                        comet_logger.log_metric(k, test_metrics[k], epoch=epoch + 1)

                # SCHEDULER
                if scheduler is not None:
                    scheduler.step(test_metrics["iou"])
                # Log learning rate to Comet.ml
                if comet_logger is not None:
                    comet_logger.log_metric("learning_rate", scheduler.get_last_lr(), epoch=epoch + 1)

def train_maskrcnn(model, device:str, dataloader: torch.utils.data.DataLoader, optimizer):
    """Training script for Mask R-CNN.

    Args:
        model: pytorch NN model.
        device: "cuda" / "cpu".
        dataloader: DataLoader for training.
        optimizer: the desired optimizer.

    Returns:
        model, 
        metrics: {"training_loss":mean_loss}
        
    """

    # Sending model to device
    model = model.to(device)

    model.train()  # Set model to training mode

    loss = 0.0
    mean_loss = 0.0

    # Iterate over the data
    for images, targets in tqdm(dataloader):
        
        # preprocessing and converting to tensors
        images = [images[0].to(device)]
        targets = [{key: targets[key].squeeze(0).to(device) for key in targets.keys()}]

        # Zero the variables that are used for storing the gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward
        with torch.set_grad_enabled(True):

            loss_dict = model(images, targets)
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

def test_maskrcnn(model:Any, device:str, dataloader:torch.utils.data.DataLoader, optimizer:Any, confidence_threshold: float=0.5):
    """Runs testing for the model.

    Args:
        model (Any): pytorch NN model.
        device (str): "cuda" / "cpu".
        dataloader (torch.utils.data.DataLoader): DataLoader for testing.
        optimizer (Any): the desired optimizer.
        confidence_threshold (float): confidence score threshold to filter predictions during evaluation.

    Returns:
        metrics: { 
                "test_loss" : float,
                "iou": float,
                "dice": float
                }
    """

    # Sending model to device
    model = model.to(device)

    metrics = { 
        "test_loss" : 0.0,
        "iou": 0.0,
        }

    loss = 0.0

    # Iterate over the data
    for images, targets in tqdm(dataloader):

        # preprocessing and converting to tensors
        images = [images[0].to(device)]
        targets = [{key: targets[key].squeeze(0).to(device) for key in targets.keys()}]
        
        # Zero the variables that are used for storing the gradients
        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(False):
            # To get training loss
            model.train()  # Set model to training mode

            loss_dict = model(images, targets)
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
            predictions = model(images)

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
                #iou_best = calculate_pixel_iou_binary_classification(prediction, target, confidence_threshold,method="best")
                iou_combined = calculate_pixel_iou_binary_classification(prediction, target, confidence_threshold,method="combined")
                metrics["iou"]+=iou_combined
                #metrics["iou_best"]+=iou_best

    
    for k in metrics.keys():
        metrics[k] = metrics[k] / len(dataloader.dataset)

    print(f"Test – IoU: {metrics['iou']:.4f}, Loss: {metrics['test_loss']:.4f}")
                    
    return metrics