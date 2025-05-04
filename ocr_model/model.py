import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import timm

class TrOCRModel(nn.Module):
    def __init__(self, char_to_idx, idx_to_char, pretrained=True):
        """
        TrOCR model for OCR.
        
        Args:
            char_to_idx: Character to index mapping
            idx_to_char: Index to character mapping
            pretrained: Whether to use pretrained weights
        """
        super(TrOCRModel, self).__init__()
        
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
        
        # Load TrOCR model
        if pretrained:
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        else:
            # Initialize from scratch
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                "microsoft/trocr-base-handwritten", "microsoft/trocr-base-handwritten"
            )
        
        # Resize token embeddings to match vocabulary size
        self.model.decoder.resize_token_embeddings(self.vocab_size)
        
        # Set special tokens
        self.model.config.decoder_start_token_id = 0  # Start token
        self.model.config.pad_token_id = 0  # Padding token
        
        # Set generation parameters
        self.model.config.eos_token_id = 1  # End of sequence token
        self.model.config.max_length = 64
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
    
    def forward(self, pixel_values, labels=None):
        """
        Forward pass.
        
        Args:
            pixel_values: Input images
            labels: Target text indices
        
        Returns:
            Model outputs
        """
        # Create decoder input ids (shifted right)
        if labels is not None:
            decoder_input_ids = self._shift_right(labels)
        else:
            batch_size = pixel_values.shape[0]
            decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=pixel_values.device) * self.model.config.decoder_start_token_id
        
        # Forward pass
        outputs = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        
        return outputs
    
    def _shift_right(self, input_ids):
        """
        Shift input ids one position to the right for teacher forcing.
        
        Args:
            input_ids: Input token indices
        
        Returns:
            Shifted input ids
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.model.config.decoder_start_token_id
        
        return shifted_input_ids
    
    def generate(self, pixel_values):
        """
        Generate text from images.
        
        Args:
            pixel_values: Input images
        
        Returns:
            Generated text
        """
        # Generate token indices
        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            max_length=self.model.config.max_length,
            num_beams=self.model.config.num_beams,
            early_stopping=self.model.config.early_stopping
        )
        
        # Convert to text
        texts = []
        for ids in generated_ids:
            text = ''.join([self.idx_to_char.get(idx.item(), '') for idx in ids])
            texts.append(text)
        
        return texts

class CRNNModel(nn.Module):
    def __init__(self, char_to_idx, idx_to_char):
        """
        CRNN model for OCR.
        
        Args:
            char_to_idx: Character to index mapping
            idx_to_char: Index to character mapping
        """
        super(CRNNModel, self).__init__()
        
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
        
        # CNN backbone (using EfficientNet)
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=1280,  # EfficientNet-B0 final feature size
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(512, self.vocab_size)  # 512 = 256*2 (bidirectional)
    
    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)[-1]  # Get last layer features
        
        # Reshape for RNN
        batch_size, channels, height, width = features.size()
        features = features.permute(0, 3, 1, 2)  # [B, W, C, H]
        features = features.reshape(batch_size, width, channels * height)
        
        # RNN sequence processing
        rnn_output, _ = self.rnn(features)
        
        # Fully connected layer
        output = self.fc(rnn_output)
        
        # Log softmax for CTC loss
        output = F.log_softmax(output, dim=2)
        
        return output
    
    def decode_ctc(self, log_probs):
        """
        Decode CTC output to text.
        
        Args:
            log_probs: Log probabilities from model output
        
        Returns:
            Decoded text
        """
        # Get most probable character at each step
        probs = torch.exp(log_probs)
        max_indices = torch.argmax(probs, dim=2)
        
        batch_size = max_indices.size(0)
        texts = []
        
        for b in range(batch_size):
            indices = max_indices[b]
            
            # Merge repeated characters
            merged_indices = []
            prev_idx = -1
            for idx in indices:
                if idx != prev_idx:
                    merged_indices.append(idx.item())
                    prev_idx = idx
            
            # Remove blank tokens (assuming 0 is blank)
            filtered_indices = [idx for idx in merged_indices if idx != 0]
            
            # Convert to text
            text = ''.join([self.idx_to_char.get(idx, '') for idx in filtered_indices])
            texts.append(text)
        
        return texts 