import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation

class LayoutSegmentationModel(nn.Module):
    def __init__(self, model_type="unet", encoder_name="resnet34", num_classes=1):
        """
        Layout segmentation model.
        
        Args:
            model_type: Type of segmentation model ('unet', 'fpn', 'segformer')
            encoder_name: Name of the encoder backbone
            num_classes: Number of output classes (1 for binary segmentation)
        """
        super(LayoutSegmentationModel, self).__init__()
        
        self.model_type = model_type
        
        if model_type == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None  # We'll apply sigmoid in forward
            )
        elif model_type == "fpn":
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None
            )
        elif model_type == "segformer":
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, x):
        if self.model_type == "segformer":
            outputs = self.model(pixel_values=x)
            logits = outputs.logits
        else:
            logits = self.model(x)
        
        # Apply sigmoid for binary segmentation
        return torch.sigmoid(logits)

def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for segmentation.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
    
    Returns:
        Dice loss
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def bce_dice_loss(pred, target, bce_weight=0.5):
    """
    Combined BCE and Dice loss.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        bce_weight: Weight for BCE loss
    
    Returns:
        Combined loss
    """
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    
    return bce_weight * bce + (1 - bce_weight) * dice 