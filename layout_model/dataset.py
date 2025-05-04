import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LayoutDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Dataset for layout segmentation.
        
        Args:
            image_dir: Directory containing document images
            mask_dir: Directory containing layout masks
            transform: Albumentations transformations
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
        
        # Get all mask files
        self.mask_files = []
        for image_path in self.image_files:
            # Extract filename without extension
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            
            # Construct mask path
            mask_filename = f"{name}_mask{ext}"
            mask_path = os.path.join(mask_dir, mask_filename)
            
            # Check if mask exists
            if os.path.exists(mask_path):
                self.mask_files.append(mask_path)
            else:
                print(f"Warning: Mask not found for {image_path}")
                # Use a placeholder mask path
                self.mask_files.append(None)
        
        # Make sure we have the same number of images and masks
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must be the same"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Get image and mask paths
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        # Skip if mask is None
        if mask_path is None:
            # Return a dummy sample
            dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
            dummy_mask = np.zeros((512, 512), dtype=np.float32)  # Explicitly use float32
            
            if self.transform:
                augmented = self.transform(image=dummy_image, mask=dummy_mask)
                return augmented['image'], augmented['mask'].float().unsqueeze(0)  # Convert to float
            return torch.from_numpy(dummy_image.transpose(2, 0, 1)), torch.from_numpy(dummy_mask[None, :, :]).float()  # Convert to float
        
        # Load image and mask
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.0  # Normalize to 0-1
            mask = mask.astype(np.float32)  # Explicitly convert to float32
        except Exception as e:
            print(f"Error loading {image_path} or {mask_path}: {e}")
            # Return a dummy sample
            dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
            dummy_mask = np.zeros((512, 512), dtype=np.float32)  # Explicitly use float32
            
            if self.transform:
                augmented = self.transform(image=dummy_image, mask=dummy_mask)
                return augmented['image'], augmented['mask'].float().unsqueeze(0)  # Convert to float
            return torch.from_numpy(dummy_image.transpose(2, 0, 1)), torch.from_numpy(dummy_mask[None, :, :]).float()  # Convert to float
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].float().unsqueeze(0)  # Convert to float
        else:
            # Convert to torch tensors
            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask[None, :, :]).float()  # Convert to float
        
        return image, mask

def get_train_transform():
    """Get training transformations with augmentations"""
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transform():
    """Get validation transformations without augmentations"""
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_dataloaders(image_dir, mask_dir, batch_size=8, val_split=0.2):
    """
    Create train and validation dataloaders.
    
    Args:
        image_dir: Directory containing document images
        mask_dir: Directory containing layout masks
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
    
    Returns:
        train_loader, val_loader
    """
    # Create dataset
    dataset = LayoutDataset(image_dir, mask_dir, transform=get_train_transform())
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Override transforms for validation dataset
    val_dataset.dataset.transform = get_val_transform()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader 