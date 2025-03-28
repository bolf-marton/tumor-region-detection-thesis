import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

def get_model_maskrcnn(num_classes, **kwargs):
    # Load the pretrained Mask R-CNN model
    dummy_model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    
    # Modify the box predictor's classification head to have the correct number of classes
    in_features = dummy_model.roi_heads.box_predictor.cls_score.in_features
    dummy_model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
    dummy_model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, num_classes * 4)
    
    # Modify the mask predictor head to have the correct number of classes
    in_channels_mask = dummy_model.roi_heads.mask_predictor.conv5_mask.in_channels
    dummy_model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(in_channels_mask, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    # Initialize the weights of the new layers
    torch.nn.init.normal_(dummy_model.roi_heads.box_predictor.cls_score.weight, mean=0.0, std=0.01)
    torch.nn.init.constant_(dummy_model.roi_heads.box_predictor.cls_score.bias, 0.0)
    torch.nn.init.normal_(dummy_model.roi_heads.box_predictor.bbox_pred.weight, mean=0.0, std=0.001)
    torch.nn.init.constant_(dummy_model.roi_heads.box_predictor.bbox_pred.bias, 0.0)
    torch.nn.init.normal_(dummy_model.roi_heads.mask_predictor.mask_fcn_logits.weight, mean=0.0, std=0.001)
    torch.nn.init.constant_(dummy_model.roi_heads.mask_predictor.mask_fcn_logits.bias, 0.0)
    
    # Save the modified state dict
    torch.save(dummy_model.state_dict(), "models/MaskRCNN_ResNet50_FPN_V2_Weights.pth")

    # Load the model with the saved state dict but without loading the unnecessary weights strictly
    model = maskrcnn_resnet50_fpn_v2(num_classes=num_classes, **kwargs)
    model.load_state_dict(torch.load("models/MaskRCNN_ResNet50_FPN_V2_Weights.pth"), strict=False)
    
    return model