import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Any
import torch.nn.functional as F

from src.utils.comet import CometLogger
from src.utils.image_tools import get_image
from src.utils.metrics import calculate_pixel_iou_binary_classification

##############
# MASK R-CNN #
##############

def maskrcnn_training_testing_loop(model:Any, device:str, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader, optimizer:Any, scheduler:Any=None, n_epochs:int=0, box_score_thresh:float=0.5, comet_logger:CometLogger=None):
    """Training loop for Mask R-CNN.
    Args:
        model (Any): pytorch NN model.
        device (str): "cuda" / "cpu".
        train_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for training.
        test_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for testing.
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
                test_metrics = test_maskrcnn(model,device,test_dataloader,optimizer,box_score_thresh)

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

            loss_weights = {
                'loss_classifier': 1.0,
                'loss_box_reg': 1.0,
                'loss_mask': 2.0,  # Emphasizing mask quality
                'loss_objectness': 1.0,
                'loss_rpn_box_reg': 1.0
            }

            # Calculate weighted sum
            loss = sum(loss_weights[k] * loss_dict[k] for k in loss_dict.keys())

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


#########
# U-NET #
#########

def unet_training_testing_loop(model:Any, device:str, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader, optimizer:Any, scheduler:Any=None, n_epochs:int=0, comet_logger:CometLogger=None):
    """Training loop for U-Net.
    Args:
        model (Any): pytorch NN model.
        device (str): "cuda" / "cpu".
        train_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for training.
        test_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for testing.
        optimizer (Any): the desired optimizer.
        scheduler (Any): the desired scheduler.
        n_epochs (int): number of epochs to train.
        comet_logger (CometLogger): Comet.ml logger (optional).
    Returns:
        model: trained model.
        metrics: { 
                "test_loss" : float,
                "iou": float,
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
                model, train_metrics = train_unet(model,device,train_dataloader,optimizer)
                
                # Log metrics to Comet.ml
                if comet_logger is not None:
                    for k in train_metrics.keys():
                        comet_logger.log_metric(k, train_metrics[k], epoch=epoch + 1)


                # TESTING
                test_metrics = test_unet(model,device,test_dataloader,optimizer)

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

def train_unet(model, device:str, dataloader: torch.utils.data.DataLoader, optimizer):
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
        image = images.to(device)
        target_masks = targets['masks'].squeeze(0)

        binary_mask = torch.zeros(
            (image.shape[0], 1, target_masks.shape[1], target_masks.shape[2]),  # Shape: [B, 1, H, W]
            device=device
        )

        # Combine instance masks into one binary mask
        if target_masks.shape[0] > 0:  # If we have any masks
            binary_mask[0] = (target_masks.sum(dim=0) > 0).float()

        # Zero the variables that are used for storing the gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward
        with torch.set_grad_enabled(True):
            # Calculate loss
            loss = F.binary_cross_entropy_with_logits(model(image), binary_mask, reduction='mean')

            mean_loss += loss

            # Backward + optimize (only if in training phase)
            loss.backward()
            optimizer.step()
    

    mean_loss = mean_loss / len(dataloader.dataset)
    
    print(f"Train – Mean Loss: {mean_loss:.4f}") # , IoU: {IoU:.4f}

    metrics = {"training_loss":mean_loss}
                
    return model, metrics

def test_unet(model: Any, device: str, dataloader: torch.utils.data.DataLoader, optimizer: Any, confidence_threshold: float = 0.5):
    """Runs testing for UNet model.

    Args:
        model: UNet model
        device: "cuda" / "cpu"
        dataloader: DataLoader for testing
        optimizer: Optimizer (needed for compatibility with training loop)
        confidence_threshold: Threshold for binary mask prediction

    Returns:
        metrics: { 
                "test_loss": float,
                "iou": float
                }
    """
    # Sending model to device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    metrics = {
        "test_loss": 0.0,
        "iou": 0.0,
    }

    # Iterate over the data
    for images, targets in tqdm(dataloader):
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
            # Forward pass
            output = model(image)
            
            # Calculate loss
            loss = F.binary_cross_entropy_with_logits(output, binary_mask, reduction='mean')
            metrics["test_loss"] += loss.item()
            
            pred_mask = (output > confidence_threshold).float()
            
            # Calculate IoU (intersection over union)
            intersection = torch.logical_and(pred_mask, binary_mask).sum().float()
            union = torch.logical_or(pred_mask, binary_mask).sum().float()
            
            # Handle edge case of empty masks
            if union > 0:
                iou = intersection / union
            else:
                iou = torch.tensor(1.0 if binary_mask.sum() == 0 else 0.0, device=device)
                
            metrics["iou"] += iou.item()

    # Average metrics over dataset
    for k in metrics.keys():
        metrics[k] = metrics[k] / len(dataloader)

    print(f"Test – IoU: {metrics['iou']:.4f}, Loss: {metrics['test_loss']:.4f}")
    
    return metrics


##########
# RESNET #
##########

def resnet_training_testing_loop(model:Any, device:str, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader, optimizer:Any, scheduler:Any=None, n_epochs:int=0, comet_logger:CometLogger=None):
    """Training loop for ResNet.
    Args:
        model (Any): pytorch NN model.
        device (str): "cuda" / "cpu".
        train_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for training.
        test_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for testing.
        optimizer (Any): the desired optimizer.
        scheduler (Any): the desired scheduler.
        n_epochs (int): number of epochs to train.
        comet_logger (CometLogger): Comet.ml logger (optional).
    Returns:
        model: trained model.
        metrics: { 
                "test_loss" : float,
                "iou": float,
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
                model, train_metrics = train_resnet(model,device,train_dataloader,optimizer)
                
                # Log metrics to Comet.ml
                if comet_logger is not None:
                    for k in train_metrics.keys():
                        comet_logger.log_metric(k, train_metrics[k], epoch=epoch + 1)


                # TESTING
                test_metrics = test_resnet(model,device,test_dataloader,optimizer)

                # Log metrics to Comet.ml
                if comet_logger is not None:
                    for k in test_metrics.keys():
                        comet_logger.log_metric(k, test_metrics[k], epoch=epoch + 1)

                # SCHEDULER
                if scheduler is not None:
                    scheduler.step(test_metrics["iou"])
                # Log learning rate to Comet.ml
                if comet_logger is not None and scheduler is not None:
                    comet_logger.log_metric("learning_rate", scheduler.get_last_lr(), epoch=epoch + 1)

def train_resnet(model, device:str, dataloader: torch.utils.data.DataLoader, optimizer):
    """Training script for ResNet classification.

    Args:
        model: pytorch ResNet model for classification.
        device: "cuda" / "cpu".
        dataloader: DataLoader for training.
        optimizer: the desired optimizer.

    Returns:
        model, 
        metrics: {"training_loss": mean_loss, "training_accuracy": accuracy}
        
    """
    # Sending model to device
    model = model.to(device)
    model.train()  # Set model to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the data
    for images, targets in tqdm(dataloader):
        # Convert images to device
        images = images.to(device)
        
        labels = targets['labels'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with torch.set_grad_enabled(True):
            # Get model outputs
            outputs = model(images)
            
            # Calculate classification loss (cross entropy)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average metrics
    mean_loss = running_loss / total
    accuracy = 100 * correct / total
    
    print(f"Train – Loss: {mean_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    metrics = {
        "training_loss": mean_loss,
        "training_accuracy": accuracy
    }
                
    return model, metrics
                        
def test_resnet(model: Any, device: str, dataloader: torch.utils.data.DataLoader, optimizer: Any):
    """Runs testing for ResNet classification model.

    Args:
        model: ResNet model
        device: "cuda" / "cpu"
        dataloader: DataLoader for testing
        optimizer: Optimizer (needed for compatibility with training loop)

    Returns:
        metrics: { 
                "test_loss": float,
                "test_accuracy": float
                }
    """ 
    # Sending model to device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the data
    for images, targets in tqdm(dataloader):
        # Convert images to device
        images = images.to(device)
        labels = targets['labels'].to(device)
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(images)
            
            # Calculate classification loss (cross entropy)
            loss = F.cross_entropy(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average metrics
    mean_loss = running_loss / total
    accuracy = 100 * correct / total
    
    print(f"Test – Loss: {mean_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    metrics = {
        "test_loss": mean_loss,
        "test_accuracy": accuracy
    }
                
    return metrics