import os
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import TrOCRModel

def load_ocr_model(model_path, device):
    """
    Load trained OCR model.
    
    Args:
        model_path: Path to the trained model
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = TrOCRModel(
        checkpoint['char_to_idx'],
        checkpoint['idx_to_char'],
        pretrained=False
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['char_to_idx'], checkpoint['idx_to_char']

def preprocess_image(image_path):
    """
    Preprocess image for OCR.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Apply transform
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)
    
    return image_tensor

def normalize_historical_text(text):
    """
    Apply normalization rules for historical Spanish text.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Replace long s (ſ) with regular s
    text = text.replace('ſ', 's')
    
    # Replace ç with z
    text = text.replace('ç', 'z')
    
    # Other normalizations
    # Remove accents except ñ
    text = ''.join(c for c in text if c not in 'áéíóúÁÉÍÓÚ')
    
    # Handle macrons (¯)
    text = text.replace('n̄', 'nn')
    text = text.replace('ā', 'a')
    text = text.replace('ē', 'e')
    text = text.replace('ī', 'i')
    text = text.replace('ō', 'o')
    text = text.replace('ū', 'u')
    
    return text

def process_regions(model, regions_dir, output_dir, device):
    """
    Process text regions with OCR model.
    
    Args:
        model: Trained OCR model
        regions_dir: Directory containing text region images
        output_dir: Directory to save OCR results
        device: Device to run inference on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all region images
    region_files = []
    for root, _, files in os.walk(regions_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                region_files.append(os.path.join(root, file))
    
    # Process each region
    results = {}
    
    for region_path in tqdm(region_files, desc="Processing regions"):
        # Preprocess image
        image_tensor = preprocess_image(region_path).to(device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(pixel_values=image_tensor)
            generated_text = model.decode(generated_ids)[0]
        
        # Normalize text
        normalized_text = normalize_historical_text(generated_text)
        
        # Extract document and page info from filename
        filename = os.path.basename(region_path)
        parts = filename.split('_')
        doc_name = parts[0]
        page_num = int(parts[2])
        region_num = int(parts[4].split('.')[0])
        
        # Create key for this document and page
        doc_page_key = f"{doc_name}_page_{page_num:03d}"
        
        # Add to results
        if doc_page_key not in results:
            results[doc_page_key] = []
        
        results[doc_page_key].append({
            'region_path': region_path,
            'region_num': region_num,
            'text': normalized_text
        })
    
    # Sort regions by region number and combine text
    for doc_page_key, regions in results.items():
        # Sort by region number
        regions.sort(key=lambda x: x['region_num'])
        
        # Combine text
        combined_text = '\n'.join([region['text'] for region in regions])
        
        # Save to file
        output_path = os.path.join(output_dir, f"{doc_page_key}_ocr.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Also save as JSON with more details
        json_path = os.path.join(output_dir, f"{doc_page_key}_ocr.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'document_page': doc_page_key,
                'combined_text': combined_text,
                'regions': regions
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {doc_page_key} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text regions with OCR model")
    parser.add_argument("--model_path", required=True, help="Path to the trained OCR model")
    parser.add_argument("--regions_dir", required=True, help="Directory containing text region images")
    parser.add_argument("--output_dir", required=True, help="Directory to save OCR results")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, char_to_idx, idx_to_char = load_ocr_model(args.model_path, device)
    
    # Process regions
    process_regions(model, args.regions_dir, args.output_dir, device) 