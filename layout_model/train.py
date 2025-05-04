import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from model import LayoutSegmentationModel, bce_dice_loss

def train_model(image_dir, mask_dir, output_dir, model_type="unet", encoder_name="resnet34", 
                batch_size=8, num_epochs=15, learning_rate=1e-4):
    """
    Train layout segmentation model.
    
    Args:
        image_dir: Directory containing document images
        mask_dir: Directory containing layout masks
        output_dir: Directory to save model checkpoints
        model_type: Type of segmentation model
        encoder_name: Name of the encoder backbone
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(image_dir, mask_dir, batch_size)
    
    # Create model
    model = LayoutSegmentationModel(model_type, encoder_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = bce_dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Update validation loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_{model_type}_{encoder_name}.pth"))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_model_{model_type}_{encoder_name}.pth"))
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train layout segmentation model")
    parser.add_argument("--image_dir", required=True, help="Directory containing document images")
    parser.add_argument("--mask_dir", required=True, help="Directory containing layout masks")
    parser.add_argument("--output_dir", required=True, help="Directory to save model checkpoints")
    parser.add_argument("--model_type", default="unet", choices=["unet", "fpn", "segformer"], help="Type of segmentation model")
    parser.add_argument("--encoder_name", default="resnet34", help="Name of the encoder backbone")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_model(
        args.image_dir,
        args.mask_dir,
        args.output_dir,
        args.model_type,
        args.encoder_name,
        args.batch_size,
        args.num_epochs,
        args.learning_rate
    ) 