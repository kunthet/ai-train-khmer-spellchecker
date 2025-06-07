"""
Character-Level LSTM Model for Khmer Language Processing

This module implements bidirectional LSTM models for character-level language
modeling of Khmer text, with attention mechanisms and Unicode-aware processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict


@dataclass
class LSTMConfiguration:
    """Configuration for LSTM model architecture and training"""
    # Model architecture
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    use_attention: bool = True
    
    # Vocabulary settings
    max_vocab_size: int = 1000
    min_char_frequency: int = 2
    sequence_length: int = 100
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 20
    patience: int = 5
    
    # Khmer-specific settings
    include_khmer_only: bool = False
    khmer_unicode_range: Tuple[int, int] = (0x1780, 0x17FF)
    special_tokens: List[str] = field(default_factory=lambda: ['<PAD>', '<UNK>', '<SOS>', '<EOS>'])


@dataclass
class ModelTrainingConfig:
    """Training configuration and hyperparameters"""
    device: str = 'cpu'  # Will be set to 'cuda' if available
    gradient_clip_norm: float = 5.0
    weight_decay: float = 1e-5
    scheduler_step_size: int = 7
    scheduler_gamma: float = 0.1
    validation_split: float = 0.2
    save_best_only: bool = True
    verbose: bool = True


class CharacterVocabulary:
    """
    Character vocabulary management for Khmer text processing
    
    Handles Unicode character mapping, special tokens, and vocabulary filtering
    for optimal LSTM training on Khmer text.
    """
    
    def __init__(self, config: LSTMConfiguration):
        self.config = config
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.char_frequencies = Counter()
        self.vocabulary_size = 0
        
        # Initialize with special tokens
        for i, token in enumerate(config.special_tokens):
            self.char_to_idx[token] = i
            self.idx_to_char[i] = token
        
        self.vocabulary_size = len(config.special_tokens)
        self.logger = logging.getLogger("character_vocabulary")
    
    def _is_khmer_character(self, char: str) -> bool:
        """Check if character is in Khmer Unicode range"""
        if not char:
            return False
        code_point = ord(char)
        return self.config.khmer_unicode_range[0] <= code_point <= self.config.khmer_unicode_range[1]
    
    def build_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """
        Build character vocabulary from texts
        
        Args:
            texts: List of text strings for vocabulary building
            
        Returns:
            Dictionary with vocabulary statistics
        """
        self.logger.info(f"Building vocabulary from {len(texts)} texts...")
        
        # Count character frequencies
        total_chars = 0
        khmer_chars = 0
        
        for text in texts:
            for char in text:
                self.char_frequencies[char] += 1
                total_chars += 1
                if self._is_khmer_character(char):
                    khmer_chars += 1
        
        # Filter vocabulary based on configuration
        filtered_chars = []
        
        for char, freq in self.char_frequencies.most_common():
            # Skip special tokens (already added)
            if char in self.config.special_tokens:
                continue
            
            # Apply frequency filter
            if freq < self.config.min_char_frequency:
                break
            
            # Apply Khmer-only filter if enabled
            if self.config.include_khmer_only and not self._is_khmer_character(char):
                continue
            
            # Check vocabulary size limit
            if len(filtered_chars) >= self.config.max_vocab_size - len(self.config.special_tokens):
                break
            
            filtered_chars.append(char)
        
        # Add filtered characters to vocabulary
        for char in filtered_chars:
            self.char_to_idx[char] = self.vocabulary_size
            self.idx_to_char[self.vocabulary_size] = char
            self.vocabulary_size += 1
        
        # Calculate statistics
        khmer_ratio = khmer_chars / total_chars if total_chars > 0 else 0
        coverage = sum(self.char_frequencies[char] for char in filtered_chars) / total_chars
        
        stats = {
            'total_characters': total_chars,
            'unique_characters': len(self.char_frequencies),
            'vocabulary_size': self.vocabulary_size,
            'khmer_characters': khmer_chars,
            'khmer_ratio': khmer_ratio,
            'coverage': coverage,
            'min_frequency': self.config.min_char_frequency,
            'filtered_characters': len(filtered_chars)
        }
        
        self.logger.info(f"Vocabulary built: {self.vocabulary_size} characters")
        self.logger.info(f"Coverage: {coverage:.1%}, Khmer ratio: {khmer_ratio:.1%}")
        
        return stats
    
    def encode_text(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to character indices
        
        Args:
            text: Input text string
            max_length: Maximum sequence length (uses config default if None)
            
        Returns:
            List of character indices
        """
        if max_length is None:
            max_length = self.config.sequence_length
        
        # Convert characters to indices
        indices = []
        unk_idx = self.char_to_idx.get('<UNK>', 1)
        
        for char in text[:max_length]:
            idx = self.char_to_idx.get(char, unk_idx)
            indices.append(idx)
        
        # Pad sequence if needed
        pad_idx = self.char_to_idx.get('<PAD>', 0)
        while len(indices) < max_length:
            indices.append(pad_idx)
        
        return indices
    
    def decode_indices(self, indices: List[int]) -> str:
        """
        Decode character indices back to text
        
        Args:
            indices: List of character indices
            
        Returns:
            Decoded text string
        """
        chars = []
        pad_idx = self.char_to_idx.get('<PAD>', 0)
        
        for idx in indices:
            if idx == pad_idx:
                break  # Stop at padding
            char = self.idx_to_char.get(idx, '<UNK>')
            if char not in self.config.special_tokens:
                chars.append(char)
        
        return ''.join(chars)
    
    def get_vocabulary_info(self) -> Dict:
        """Get comprehensive vocabulary information"""
        khmer_chars = [char for char in self.char_to_idx.keys() 
                      if char not in self.config.special_tokens and self._is_khmer_character(char)]
        
        return {
            'vocabulary_size': self.vocabulary_size,
            'special_tokens': self.config.special_tokens,
            'khmer_characters': len(khmer_chars),
            'total_frequency': sum(self.char_frequencies.values()),
            'most_common_chars': self.char_frequencies.most_common(20),
            'sample_khmer_chars': khmer_chars[:20]
        }
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'config': {
                'max_vocab_size': self.config.max_vocab_size,
                'min_char_frequency': self.config.min_char_frequency,
                'khmer_unicode_range': self.config.khmer_unicode_range,
                'special_tokens': self.config.special_tokens
            },
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'char_frequencies': dict(self.char_frequencies),
            'vocabulary_size': self.vocabulary_size
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(vocab_data, f)
        
        self.logger.info(f"Vocabulary saved to: {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from file"""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                vocab_data = pickle.load(f)
        
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        self.char_frequencies = Counter(vocab_data['char_frequencies'])
        self.vocabulary_size = vocab_data['vocabulary_size']
        
        self.logger.info(f"Vocabulary loaded from: {filepath}")


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM hidden states
    
    Provides weighted attention over sequence positions for better
    context understanding in character-level modeling.
    """
    
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM outputs
        
        Args:
            lstm_outputs: LSTM hidden states [batch_size, seq_len, hidden_dim]
            mask: Optional padding mask [batch_size, seq_len]
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_outputs).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Calculate attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Apply attention to get weighted representation
        attended_output = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        
        return attended_output, attention_weights


class CharacterLSTMModel(nn.Module):
    """
    Bidirectional LSTM model for character-level Khmer language modeling
    
    Features:
    - Bidirectional LSTM for forward and backward context
    - Optional attention mechanism for sequence representation
    - Character-level embedding and prediction
    - Khmer Unicode-aware processing
    """
    
    def __init__(self, config: LSTMConfiguration, vocabulary_size: int):
        super(CharacterLSTMModel, self).__init__()
        self.config = config
        self.vocabulary_size = vocabulary_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # Attention layer (optional)
        if config.use_attention:
            self.attention = AttentionLayer(lstm_output_dim)
            output_dim = lstm_output_dim
        else:
            self.attention = None
            output_dim = lstm_output_dim
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(output_dim, vocabulary_size)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger = logging.getLogger("character_lstm_model")
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Character indices [batch_size, seq_len]
            attention_mask: Padding mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Logits [batch_size, vocabulary_size] or tuple with attention weights
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM
        lstm_outputs, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim * directions]
        
        # Apply attention or use last hidden state
        if self.attention is not None:
            sequence_output, attention_weights = self.attention(lstm_outputs, attention_mask)
        else:
            # Use last non-padded hidden state
            if attention_mask is not None:
                # Get lengths of sequences
                lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                sequence_output = lstm_outputs[range(batch_size), lengths]
            else:
                sequence_output = lstm_outputs[:, -1, :]  # Last hidden state
            attention_weights = None
        
        # Apply dropout and project to vocabulary
        sequence_output = self.dropout(sequence_output)
        logits = self.output_projection(sequence_output)  # [batch_size, vocabulary_size]
        
        if return_attention and attention_weights is not None:
            return logits, attention_weights
        else:
            return logits
    
    def predict_next_character(self, 
                             input_sequence: torch.Tensor, 
                             temperature: float = 1.0,
                             top_k: Optional[int] = None) -> Tuple[int, float]:
        """
        Predict next character given input sequence
        
        Args:
            input_sequence: Character indices [1, seq_len]
            temperature: Sampling temperature for diversity
            top_k: Limit predictions to top-k most likely characters
            
        Returns:
            Tuple of (predicted_char_idx, confidence_score)
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            logits = self.forward(input_sequence)  # [1, vocabulary_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits_filtered = torch.full_like(logits, -float('inf'))
                logits_filtered.scatter_(1, indices, values)
                logits = logits_filtered
            
            # Calculate probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted_idx].item()
            
            return predicted_idx, confidence
    
    def calculate_perplexity(self, 
                           input_sequences: torch.Tensor, 
                           target_sequences: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> float:
        """
        Calculate perplexity on given sequences
        
        Args:
            input_sequences: Input character indices [batch_size, seq_len]
            target_sequences: Target character indices [batch_size, seq_len]
            attention_mask: Padding mask [batch_size, seq_len]
            
        Returns:
            Perplexity score
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            logits = self.forward(input_sequences, attention_mask)
            
            # Calculate cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
            
            # Reshape for loss calculation
            batch_size, seq_len = target_sequences.size()
            logits_flat = logits.view(-1, self.vocabulary_size)
            targets_flat = target_sequences.view(-1)
            
            # Calculate loss
            loss = loss_fn(logits_flat, targets_flat)
            
            # Convert to perplexity
            perplexity = torch.exp(loss).item()
            
            return perplexity
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CharacterLSTMModel',
            'vocabulary_size': self.vocabulary_size,
            'embedding_dim': self.config.embedding_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'bidirectional': self.config.bidirectional,
            'use_attention': self.config.use_attention,
            'dropout': self.config.dropout,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def save_model(self, filepath: str, vocabulary: CharacterVocabulary):
        """Save complete model with configuration and vocabulary"""
        model_data = {
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.config.embedding_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout,
                'bidirectional': self.config.bidirectional,
                'use_attention': self.config.use_attention,
                'max_vocab_size': self.config.max_vocab_size,
                'sequence_length': self.config.sequence_length
            },
            'vocabulary_size': self.vocabulary_size,
            'model_info': self.get_model_info()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_data, filepath)
        
        # Save vocabulary separately
        vocab_path = filepath.replace('.pth', '_vocab.json')
        vocabulary.save_vocabulary(vocab_path)
        
        self.logger.info(f"Model saved to: {filepath}")
        self.logger.info(f"Vocabulary saved to: {vocab_path}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu') -> Tuple['CharacterLSTMModel', CharacterVocabulary]:
        """Load complete model with configuration and vocabulary"""
        # Load model data
        model_data = torch.load(filepath, map_location=device)
        
        # Reconstruct configuration
        config = LSTMConfiguration(
            embedding_dim=model_data['config']['embedding_dim'],
            hidden_dim=model_data['config']['hidden_dim'],
            num_layers=model_data['config']['num_layers'],
            dropout=model_data['config']['dropout'],
            bidirectional=model_data['config']['bidirectional'],
            use_attention=model_data['config']['use_attention'],
            max_vocab_size=model_data['config']['max_vocab_size'],
            sequence_length=model_data['config']['sequence_length']
        )
        
        # Create model
        model = cls(config, model_data['vocabulary_size'])
        model.load_state_dict(model_data['model_state_dict'])
        model.to(device)
        
        # Load vocabulary
        vocab_path = filepath.replace('.pth', '_vocab.json')
        vocabulary = CharacterVocabulary(config)
        vocabulary.load_vocabulary(vocab_path)
        
        return model, vocabulary


if __name__ == "__main__":
    # Demo usage
    print("🔧 CHARACTER LSTM MODEL DEMO")
    print("=" * 30)
    
    # Create configuration
    config = LSTMConfiguration(
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        use_attention=True,
        max_vocab_size=500,
        sequence_length=50
    )
    
    print(f"Configuration: {config.embedding_dim}D embeddings, {config.hidden_dim}D hidden")
    
    # Sample texts for vocabulary building
    sample_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
        "ភាសាខ្មែរមានលក្ខណៈពិសេស។",
        "ការអប់រំជាមូលដ្ឋានការអភិវឌ្ឍន៍។"
    ]
    
    print(f"Building vocabulary from {len(sample_texts)} sample texts...")
    
    # Build vocabulary
    vocabulary = CharacterVocabulary(config)
    vocab_stats = vocabulary.build_vocabulary(sample_texts)
    
    print(f"✅ Vocabulary: {vocab_stats['vocabulary_size']} characters")
    print(f"   Coverage: {vocab_stats['coverage']:.1%}")
    print(f"   Khmer ratio: {vocab_stats['khmer_ratio']:.1%}")
    
    # Create model
    model = CharacterLSTMModel(config, vocabulary.vocabulary_size)
    model_info = model.get_model_info()
    
    print(f"✅ Model: {model_info['total_parameters']:,} parameters")
    print(f"   Size: {model_info['model_size_mb']:.1f} MB")
    print(f"   Bidirectional: {model_info['bidirectional']}")
    print(f"   Attention: {model_info['use_attention']}")
    
    # Test encoding/decoding
    test_text = "នេះជាការសាកល្បង"
    encoded = vocabulary.encode_text(test_text, max_length=20)
    decoded = vocabulary.decode_indices(encoded)
    
    print(f"\n🧪 Text encoding test:")
    print(f"   Original: '{test_text}'")
    print(f"   Encoded: {encoded[:10]}...")
    print(f"   Decoded: '{decoded}'")
    print(f"   Match: {'✅' if test_text in decoded else '❌'}")
    
    # Test model forward pass
    input_tensor = torch.tensor([encoded]).long()
    with torch.no_grad():
        output = model(input_tensor)
        print(f"\n🔮 Model forward pass:")
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n✅ Character LSTM model demo completed!") 