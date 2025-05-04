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

from dataset import get_dataloaders
from model import TrOCRModel, CRNNModel

def train_trocr_model(regions_dir, transcriptions_dir, output_dir, batch_size=8, num_epochs=5, learning_rate=5e-5):
    """
    Train TrOCR model.
    
    Args:
        regions_dir: Directory containing text region images
        transcriptions_dir: Directory containing transcription JSON files
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, val_loader, char_to_idx, idx_to_char = get_dataloaders(
        regions_dir, 
        transcriptions_dir, 
        batch_size=batch_size
    )
    
    # Create model
    model = TrOCRModel(char_to_idx, idx_to_char)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, encoded_texts, texts, doc_page_keys, region_nums) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            # Move data to device
            images = images.to(device)
            encoded_texts = encoded_texts.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Get logits from model output
            if isinstance(outputs, dict):
                logits = outputs.logits
            else:
                logits = outputs
                
            # Calculate input and target lengths
            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(text.strip()) for text in texts], dtype=torch.long, device=device)
            
            # Make sure target lengths are not zero
            target_lengths = torch.clamp(target_lengths, min=1)
            
            # Calculate loss
            loss = criterion(logits.transpose(0, 1), encoded_texts, input_lengths, target_lengths)
            
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
            for batch_idx, (images, encoded_texts, texts, doc_page_keys, region_nums) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
                # Move data to device
                images = images.to(device)
                encoded_texts = encoded_texts.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Get logits from model output
                if isinstance(outputs, dict):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Calculate input and target lengths
                input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device)
                target_lengths = torch.tensor([len(text.strip()) for text in texts], dtype=torch.long, device=device)
                
                # Make sure target lengths are not zero
                target_lengths = torch.clamp(target_lengths, min=1)
                
                # Calculate loss
                loss = criterion(logits.transpose(0, 1), encoded_texts, input_lengths, target_lengths)
                
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'epoch': epoch,
                'loss': best_val_loss
            }, os.path.join(output_dir, "best_trocr_model.pth"))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    
    return model, char_to_idx, idx_to_char

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR model")
    parser.add_argument("--regions_dir", required=True, help="Directory containing text region images")
    parser.add_argument("--transcriptions_dir", required=True, help="Directory containing transcription JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()
    
    train_trocr_model(
        args.regions_dir,
        args.transcriptions_dir,
        args.output_dir,
        args.batch_size,
        args.num_epochs,
        args.learning_rate
    ) 