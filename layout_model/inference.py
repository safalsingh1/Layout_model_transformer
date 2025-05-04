import os
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

from model import LayoutSegmentationModel

def predict_layout(model, image_path, device, threshold=0.5):
    """
    Predict layout mask for a single image.
    
    Args:
        model: Trained layout segmentation model
        image_path: Path to the input image
        device: Device to run inference on
        threshold: Threshold for binary segmentation
    
    Returns:
        Predicted mask as numpy array
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define transform
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Apply transform
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        output = output.squeeze().cpu().numpy()
    
    # Apply threshold
    mask = (output > threshold).astype(np.uint8)
    
    # Resize back to original size
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask

def extract_text_regions(image, mask):
    """
    Extract text regions from image based on mask.
    
    Args:
        image: Input image as numpy array
        mask: Binary mask as numpy array
    
    Returns:
        List of cropped text regions
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by y-coordinate (top to bottom)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    text_regions = []
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out small regions
        if w < 50 or h < 50:
            continue
        
        # Crop region from image
        region = image[y:y+h, x:x+w]
        
        text_regions.append({
            'region': region,
            'bbox': (x, y, w, h)
        })
    
    return text_regions

def process_document_images(model_path, image_dir, output_dir, model_type="unet", encoder_name="resnet34"):
    """
    Process document images with layout model.
    
    Args:
        model_path: Path to the trained model
        image_dir: Directory containing document images
        output_dir: Directory to save results
        model_type: Type of segmentation model
        encoder_name: Name of the encoder backbone
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "regions"), exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutSegmentationModel(model_type, encoder_name, num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        # Get image filename
        image_filename = os.path.basename(image_path)
        
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}, skipping...")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
        
        # Predict layout mask
        mask = predict_layout(model, image_path, device)
        
        # Save mask
        mask_path = os.path.join(output_dir, "masks", image_filename)
        cv2.imwrite(mask_path, mask * 255)
        
        # Extract text regions
        text_regions = extract_text_regions(image_rgb, mask)
        
        # Save text regions
        regions_dir = os.path.join(output_dir, "regions", os.path.splitext(image_filename)[0])
        os.makedirs(regions_dir, exist_ok=True)
        
        regions_data = []
        for i, region in enumerate(text_regions):
            region_path = os.path.join(regions_dir, f"region_{i:03d}.png")
            cv2.imwrite(region_path, region['region'])
            
            # Get bounding box
            bbox = region['bbox']
            
            # Add to regions data
            regions_data.append({
                'region_id': i,
                'region_path': region_path,
                'bbox': bbox
            })
        
        # Save regions data as JSON
        json_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_regions.json")
        with open(json_path, 'w') as f:
            json.dump({
                'image_path': image_path,
                'mask_path': mask_path,
                'regions': regions_data
            }, f, indent=2)
        
        print(f"Processed {image_path} -> {len(text_regions)} text regions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process document images with layout model")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--image_dir", required=True, help="Directory containing document images")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--model_type", default="unet", choices=["unet", "fpn", "segformer"], help="Type of segmentation model")
    parser.add_argument("--encoder_name", default="resnet34", help="Name of the encoder backbone")
    args = parser.parse_args()
    
    process_document_images(
        args.model_path,
        args.image_dir,
        args.output_dir,
        args.model_type,
        args.encoder_name
    ) 