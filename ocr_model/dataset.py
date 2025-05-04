import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OCRDataset(Dataset):
    def __init__(self, regions_dir, transcriptions_dir, transform=None, max_length=128):
        """
        Dataset for OCR model.
        
        Args:
            regions_dir: Directory containing text region images
            transcriptions_dir: Directory containing transcription JSON files
            transform: Albumentations transformations
            max_length: Maximum length of text sequence
        """
        self.regions_dir = regions_dir
        self.transcriptions_dir = transcriptions_dir
        self.transform = transform
        self.max_length = max_length
        
        # Get all region images
        self.region_files = []
        for root, _, files in os.walk(regions_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.region_files.append(os.path.join(root, file))
        
        # Sort for reproducibility
        self.region_files.sort()
        
        # Load transcriptions
        self.transcriptions = self._load_transcriptions()
        
        # Create character to index mapping
        self.char_to_idx = {' ': 0}  # Start with space
        for i, c in enumerate(
            "abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ0123456789"
            ".,;:!?¿¡()[]{}-_'\"+=*/\\@#$%^&<>|~`áéíóúÁÉÍÓÚüÜ"
        ):
            self.char_to_idx[c] = i + 1
        
        # Add special tokens
        self.char_to_idx['<PAD>'] = len(self.char_to_idx)
        self.char_to_idx['<UNK>'] = len(self.char_to_idx)
        self.char_to_idx['<SOS>'] = len(self.char_to_idx)
        self.char_to_idx['<EOS>'] = len(self.char_to_idx)
        
        # Create index to character mapping
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def _load_transcriptions(self):
        """Load transcriptions from JSON files"""
        transcriptions = {}
        
        # Get all JSON files
        json_files = []
        for root, _, files in os.walk(self.transcriptions_dir):
            for file in files:
                if file.lower().endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        # Load each JSON file
        for json_path in json_files:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract document name
                doc_name = os.path.basename(json_path).split('_')[0]
                
                # Get transcriptions for each page
                if 'pages' in data:
                    for i, page_text in enumerate(data['pages']):
                        # Create key for this page
                        page_key = f"{doc_name}_page_{i+1:03d}"
                        
                        # Store transcription
                        transcriptions[page_key] = page_text
        
        return transcriptions
    
    def _normalize_text(self, text):
        """Apply normalization rules for historical Spanish text"""
        # Replace long s (ſ) with regular s
        text = text.replace('ſ', 's')
        
        # Replace ç with z
        text = text.replace('ç', 'z')
        
        # Other normalizations as needed
        
        return text
    
    def _text_to_indices(self, text):
        """Convert text to sequence of indices"""
        # Normalize text
        text = self._normalize_text(text)
        
        # Convert to indices
        indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in text]
        
        # Add start and end tokens
        indices = [self.char_to_idx['<SOS>']] + indices + [self.char_to_idx['<EOS>']]
        
        # Pad or truncate to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.char_to_idx['<PAD>']] * (self.max_length - len(indices))
        
        return torch.tensor(indices)
    
    def __len__(self):
        return len(self.region_files)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        region_path = self.region_files[idx]
        
        # Extract document and page info from region path
        # Expected format: path/to/regions/document_name/region_XXX.png
        try:
            # Get the parent directory name (document_name)
            parent_dir = os.path.basename(os.path.dirname(region_path))
            
            # Get region number from filename (region_XXX.png)
            region_filename = os.path.basename(region_path)
            region_num = int(region_filename.split('_')[1].split('.')[0])
            
            # Try to extract document and page info
            parts = parent_dir.split('_')
            if len(parts) >= 3:
                # If filename has expected format: document_name_pageXXX
                doc_name = '_'.join(parts[:-1])
                page_num = int(parts[-1].replace('page', ''))
            else:
                # If filename doesn't match expected format, use simpler approach
                doc_name = parent_dir
                page_num = 1  # Default to page 1
        except Exception as e:
            print(f"Warning: Could not parse document info from {region_path}: {e}")
            # Use fallback values
            doc_name = "unknown"
            page_num = 1
            region_num = idx
        
        # Create a key for this document page
        doc_page_key = f"{doc_name}_page{page_num}"
        
        # Load region image
        try:
            image = Image.open(region_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {region_path}: {e}")
            # Return a dummy sample
            dummy_image = np.zeros((384, 384, 3), dtype=np.uint8)
            dummy_text = ""
            dummy_encoded = torch.zeros(self.max_length, dtype=torch.long)
            
            if self.transform:
                augmented = self.transform(image=dummy_image)
                return augmented['image'], dummy_encoded, dummy_text, doc_page_key, region_num
            
            return torch.from_numpy(dummy_image.transpose(2, 0, 1)), dummy_encoded, dummy_text, doc_page_key, region_num
        
        # Get transcription for this document page
        transcription = self.transcriptions.get(doc_page_key, "")
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Convert to torch tensor
            image = torch.from_numpy(image.transpose(2, 0, 1))
        
        # Encode text
        encoded_text = self._text_to_indices(transcription)
        
        return image, encoded_text, transcription, doc_page_key, region_num

def get_train_transform():
    """Get training transformations"""
    return A.Compose([
        A.Resize(384, 384),  # Change from 224x224 to 384x384
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
    """Get validation transformations"""
    return A.Compose([
        A.Resize(384, 384),  # Change from 224x224 to 384x384
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_dataloaders(regions_dir, transcriptions_dir, batch_size=8, val_split=0.2):
    """
    Create train and validation dataloaders.
    
    Args:
        regions_dir: Directory containing text region images
        transcriptions_dir: Directory containing transcription JSON files
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
    
    Returns:
        train_loader, val_loader, char_to_idx, idx_to_char
    """
    # Create dataset
    dataset = OCRDataset(regions_dir, transcriptions_dir, transform=get_train_transform())
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Override transforms for validation dataset
    val_dataset.dataset.transform = get_val_transform()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, dataset.char_to_idx, dataset.idx_to_char 