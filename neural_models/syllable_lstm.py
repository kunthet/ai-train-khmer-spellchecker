"""
Syllable-Level LSTM Models for Khmer Spellchecking

This module provides syllable-level neural models that use Khmer syllable segmentation
for tokenization, offering more linguistically appropriate processing compared to
character-level models.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import existing syllable segmentation
from word_cluster.subword_cluster import khmer_syllables_no_regex_fast
from word_cluster.syllable_api import SyllableSegmentationAPI, SegmentationMethod


@dataclass
class SyllableLSTMConfiguration:
    """Configuration for syllable-level LSTM models"""
    # Model architecture
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    use_attention: bool = True
    
    # Vocabulary settings
    max_vocab_size: int = 8000
    min_syllable_frequency: int = 2
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"
    start_token: str = "<START>"
    end_token: str = "<END>"
    
    # Training settings
    sequence_length: int = 20  # Number of syllables
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 20
    patience: int = 5


class SyllableVocabulary:
    """
    Syllable-level vocabulary for Khmer neural models
    
    Uses existing syllable segmentation to build vocabulary from Khmer text corpus,
    providing more linguistically meaningful tokenization than character-level.
    """
    
    def __init__(self, config: SyllableLSTMConfiguration):
        self.config = config
        self.syllable_to_id: Dict[str, int] = {}
        self.id_to_syllable: Dict[int, str] = {}
        self.syllable_frequencies: Dict[str, int] = {}
        self.vocabulary_size = 0
        
        # Initialize segmentation API
        self.segmentation_api = SyllableSegmentationAPI(SegmentationMethod.NO_REGEX_FAST)
        
        self.logger = logging.getLogger("syllable_vocabulary")
        
        # Reserve special token IDs
        self._initialize_special_tokens()
    
    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        special_tokens = [
            self.config.pad_token,
            self.config.unk_token, 
            self.config.start_token,
            self.config.end_token
        ]
        
        for i, token in enumerate(special_tokens):
            self.syllable_to_id[token] = i
            self.id_to_syllable[i] = token
        
        self.vocabulary_size = len(special_tokens)
    
    def build_vocabulary(self, texts: List[str]) -> Dict[str, Any]:
        """
        Build syllable vocabulary from training texts
        
        Args:
            texts: List of Khmer training texts
            
        Returns:
            Dictionary with vocabulary statistics
        """
        self.logger.info(f"Building syllable vocabulary from {len(texts)} texts...")
        
        start_time = time.time()
        syllable_counter = Counter()
        total_syllables = 0
        total_chars = 0
        khmer_syllables = 0
        
        # Collect syllables from all texts
        for text in texts:
            try:
                # Segment text into syllables
                result = self.segmentation_api.segment_text(text)
                if result.success:
                    syllables = result.syllables
                    
                    for syllable in syllables:
                        # Count syllable frequency
                        syllable_counter[syllable] += 1
                        total_syllables += 1
                        total_chars += len(syllable)
                        
                        # Check if it's a Khmer syllable
                        if self._is_khmer_syllable(syllable):
                            khmer_syllables += 1
                
            except Exception as e:
                self.logger.warning(f"Error processing text: {e}")
                continue
        
        # Build vocabulary from frequent syllables
        frequent_syllables = [
            syllable for syllable, freq in syllable_counter.most_common()
            if freq >= self.config.min_syllable_frequency
        ]
        
        # Limit vocabulary size
        if len(frequent_syllables) > self.config.max_vocab_size - 4:  # Account for special tokens
            frequent_syllables = frequent_syllables[:self.config.max_vocab_size - 4]
        
        # Add to vocabulary
        for syllable in frequent_syllables:
            if syllable not in self.syllable_to_id:
                self.syllable_to_id[syllable] = self.vocabulary_size
                self.id_to_syllable[self.vocabulary_size] = syllable
                self.vocabulary_size += 1
        
        # Store frequencies
        self.syllable_frequencies = dict(syllable_counter)
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        unique_syllables = len(syllable_counter)
        khmer_ratio = khmer_syllables / total_syllables if total_syllables > 0 else 0
        coverage = len(frequent_syllables) / unique_syllables if unique_syllables > 0 else 0
        
        stats = {
            'vocabulary_size': self.vocabulary_size,
            'total_syllables': total_syllables,
            'unique_syllables': unique_syllables,
            'khmer_ratio': khmer_ratio,
            'coverage': coverage,
            'avg_syllable_length': total_chars / total_syllables if total_syllables > 0 else 0,
            'processing_time': processing_time
        }
        
        self.logger.info(f"Vocabulary built: {self.vocabulary_size} syllables")
        self.logger.info(f"Coverage: {coverage:.1%} of unique syllables")
        self.logger.info(f"Khmer ratio: {khmer_ratio:.1%}")
        
        return stats
    
    def _is_khmer_syllable(self, syllable: str) -> bool:
        """Check if syllable contains primarily Khmer characters"""
        if not syllable.strip():
            return False
        
        khmer_chars = sum(1 for char in syllable if 0x1780 <= ord(char) <= 0x17FF)
        return khmer_chars / len(syllable) >= 0.5
    
    def encode_text(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to syllable IDs
        
        Args:
            text: Input Khmer text
            max_length: Maximum sequence length (pad/truncate)
            
        Returns:
            List of syllable IDs
        """
        try:
            # Segment text
            result = self.segmentation_api.segment_text(text)
            if not result.success:
                return []
            
            # Convert syllables to IDs
            syllable_ids = []
            for syllable in result.syllables:
                syllable_id = self.syllable_to_id.get(
                    syllable, 
                    self.syllable_to_id[self.config.unk_token]
                )
                syllable_ids.append(syllable_id)
            
            # Apply length constraints
            if max_length:
                if len(syllable_ids) > max_length:
                    syllable_ids = syllable_ids[:max_length]
                else:
                    # Pad with PAD tokens
                    pad_id = self.syllable_to_id[self.config.pad_token]
                    syllable_ids.extend([pad_id] * (max_length - len(syllable_ids)))
            
            return syllable_ids
            
        except Exception as e:
            self.logger.error(f"Error encoding text: {e}")
            return []
    
    def decode_ids(self, syllable_ids: List[int]) -> str:
        """
        Decode syllable IDs back to text
        
        Args:
            syllable_ids: List of syllable IDs
            
        Returns:
            Decoded text
        """
        syllables = []
        for syllable_id in syllable_ids:
            if syllable_id in self.id_to_syllable:
                syllable = self.id_to_syllable[syllable_id]
                # Skip special tokens except spaces
                if syllable not in [self.config.pad_token, self.config.start_token, self.config.end_token]:
                    syllables.append(syllable)
        
        # Join syllables (they may contain spaces already)
        return ''.join(syllables)
    
    def get_syllable_info(self, syllable: str) -> Dict[str, Any]:
        """Get information about a specific syllable"""
        return {
            'syllable': syllable,
            'id': self.syllable_to_id.get(syllable, -1),
            'frequency': self.syllable_frequencies.get(syllable, 0),
            'in_vocabulary': syllable in self.syllable_to_id,
            'is_khmer': self._is_khmer_syllable(syllable)
        }
    
    def get_vocabulary_info(self) -> Dict[str, Any]:
        """Get comprehensive vocabulary information"""
        # Calculate syllable length distribution
        syllable_lengths = {}
        khmer_syllables = 0
        
        for syllable in self.syllable_to_id.keys():
            if syllable not in [self.config.pad_token, self.config.unk_token, 
                              self.config.start_token, self.config.end_token]:
                length = len(syllable)
                syllable_lengths[length] = syllable_lengths.get(length, 0) + 1
                
                if self._is_khmer_syllable(syllable):
                    khmer_syllables += 1
        
        return {
            'vocabulary_size': self.vocabulary_size,
            'khmer_syllables': khmer_syllables,
            'khmer_ratio': khmer_syllables / max(1, self.vocabulary_size - 4),
            'syllable_lengths': syllable_lengths,
            'avg_syllable_length': np.mean([len(s) for s in self.syllable_to_id.keys() 
                                          if s not in [self.config.pad_token, self.config.unk_token, 
                                                     self.config.start_token, self.config.end_token]]),
            'most_frequent': list(Counter(self.syllable_frequencies).most_common(10))
        }
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'config': {
                'max_vocab_size': self.config.max_vocab_size,
                'min_syllable_frequency': self.config.min_syllable_frequency,
                'special_tokens': {
                    'unk': self.config.unk_token,
                    'pad': self.config.pad_token,
                    'start': self.config.start_token,
                    'end': self.config.end_token
                }
            },
            'syllable_to_id': self.syllable_to_id,
            'id_to_syllable': self.id_to_syllable,
            'syllable_frequencies': self.syllable_frequencies,
            'vocabulary_size': self.vocabulary_size
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.syllable_to_id = vocab_data['syllable_to_id']
        self.id_to_syllable = {int(k): v for k, v in vocab_data['id_to_syllable'].items()}
        self.syllable_frequencies = vocab_data['syllable_frequencies']
        self.vocabulary_size = vocab_data['vocabulary_size']


class SyllableLSTMModel(nn.Module):
    """
    Syllable-level LSTM model for Khmer spellchecking
    
    Uses syllable embeddings and LSTM layers to model syllable-to-syllable
    transitions, providing more linguistically appropriate modeling than
    character-level approaches.
    """
    
    def __init__(self, config: SyllableLSTMConfiguration, vocab_size: int):
        super(SyllableLSTMModel, self).__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        
        # Syllable embedding layer
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # Attention mechanism (optional)
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True
            )
        
        # Output projection
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(lstm_output_size, vocab_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        # Embedding weights
        nn.init.normal_(self.embedding.weight, 0.0, 0.1)
        
        # LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            input_ids: Tensor of syllable IDs [batch_size, seq_len]
            lengths: Tensor of sequence lengths (optional)
            
        Returns:
            Logits tensor [batch_size, vocab_size] for next syllable prediction
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM
        if lengths is not None:
            # Pack padded sequences for efficiency
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Use the last timestep output for prediction
        if self.config.bidirectional:
            # Concatenate forward and backward hidden states
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Apply attention if configured
        if self.config.use_attention:
            # Use last hidden state as query
            query = last_hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]
            attended_output, _ = self.attention(query, lstm_output, lstm_output)
            output = attended_output.squeeze(1)  # [batch_size, hidden_size]
        else:
            output = last_hidden
        
        # Apply dropout and output projection
        output = self.dropout(output)
        logits = self.output_projection(output)  # [batch_size, vocab_size]
        
        return logits
    
    def predict_next_syllable(self, input_ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Predict next syllable with temperature sampling
        
        Args:
            input_ids: Input syllable sequence
            temperature: Sampling temperature (1.0 = no change, <1.0 = more focused)
            
        Returns:
            Predicted syllable ID
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from distribution
            predicted_id = torch.multinomial(probs, 1)
            
            return predicted_id
    
    def get_syllable_perplexity(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Calculate perplexity for syllable sequences"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, target_ids)
            perplexity = torch.exp(loss).item()
            return perplexity
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'SyllableLSTM',
            'vocabulary_size': self.vocab_size,
            'embedding_dim': self.config.embedding_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'bidirectional': self.config.bidirectional,
            'use_attention': self.config.use_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assume float32
        }
    
    def save_model(self, filepath: str, vocabulary: SyllableVocabulary):
        """Save model and vocabulary"""
        save_data = {
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.config.embedding_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout,
                'bidirectional': self.config.bidirectional,
                'use_attention': self.config.use_attention,
                'vocab_size': self.vocab_size
            },
            'vocabulary': {
                'syllable_to_id': vocabulary.syllable_to_id,
                'id_to_syllable': vocabulary.id_to_syllable,
                'vocabulary_size': vocabulary.vocabulary_size
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> Tuple['SyllableLSTMModel', 'SyllableVocabulary']:
        """Load model and vocabulary"""
        save_data = torch.load(filepath, map_location='cpu')
        
        # Reconstruct configuration
        config = SyllableLSTMConfiguration()
        config.embedding_dim = save_data['config']['embedding_dim']
        config.hidden_dim = save_data['config']['hidden_dim']
        config.num_layers = save_data['config']['num_layers']
        config.dropout = save_data['config']['dropout']
        config.bidirectional = save_data['config']['bidirectional']
        config.use_attention = save_data['config']['use_attention']
        
        # Create model and load weights
        model = cls(config, save_data['config']['vocab_size'])
        model.load_state_dict(save_data['model_state_dict'])
        
        # Reconstruct vocabulary
        vocabulary = SyllableVocabulary(config)
        vocabulary.syllable_to_id = save_data['vocabulary']['syllable_to_id']
        vocabulary.id_to_syllable = save_data['vocabulary']['id_to_syllable']
        vocabulary.vocabulary_size = save_data['vocabulary']['vocabulary_size']
        
        return model, vocabulary


if __name__ == "__main__":
    # Demo usage
    import time
    
    print("🔤 SYLLABLE LSTM MODEL DEMO")
    print("=" * 30)
    
    # Configuration
    config = SyllableLSTMConfiguration(
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        sequence_length=15,
        max_vocab_size=1000
    )
    
    print(f"Config: {config.embedding_dim}D embeddings, {config.hidden_dim}D hidden")
    print(f"Max vocab: {config.max_vocab_size} syllables")
    
    # Sample texts
    sample_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរដ៏វែងសម្រាប់ការបណ្តុះបណ្តាលម៉ូដែលនៅកម្រិតព្យាង្គ។",
        "ភាសាខ្មែរមានលក្ខណៈពិសេសនិងស្រស់ស្អាតណាស់។",
        "ការអប់រំជាមូលដ្ឋានសំខាន់សម្រាប់ការអភិវឌ្ឍន៍ប្រទេសជាតិ។",
        "វប្បធម៌ខ្មែរមានប្រវត្តិដ៏យូរលង់និងសម្បូរបែប។",
        "កម្ពុជាជាប្រទេសដែលមានធនធានធម្មជាតិច្រើនប្រភេទ។"
    ]
    
    print(f"Sample texts: {len(sample_texts)} training samples")
    
    # Build vocabulary
    vocabulary = SyllableVocabulary(config)
    vocab_stats = vocabulary.build_vocabulary(sample_texts)
    
    print(f"✅ Vocabulary: {vocab_stats['vocabulary_size']} syllables")
    print(f"   Coverage: {vocab_stats['coverage']:.1%}")
    print(f"   Khmer ratio: {vocab_stats['khmer_ratio']:.1%}")
    print(f"   Avg syllable length: {vocab_stats['avg_syllable_length']:.1f}")
    
    # Create model
    model = SyllableLSTMModel(config, vocabulary.vocabulary_size)
    model_info = model.get_model_info()
    
    print(f"✅ Model created:")
    print(f"   Parameters: {model_info['total_parameters']:,}")
    print(f"   Size: {model_info['model_size_mb']:.1f} MB")
    print(f"   Bidirectional: {model_info['bidirectional']}")
    print(f"   Attention: {model_info['use_attention']}")
    
    # Test encoding/decoding
    test_text = "នេះជាការសាកល្បង"
    encoded = vocabulary.encode_text(test_text, max_length=10)
    decoded = vocabulary.decode_ids(encoded)
    
    print(f"✅ Encoding test:")
    print(f"   Original: '{test_text}'")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: '{decoded}'")
    print(f"   Match: {'✅' if test_text in decoded else '❌'}")
    
    # Test forward pass
    input_tensor = torch.tensor([encoded]).long()
    with torch.no_grad():
        output = model(input_tensor)
        output_shape = output.shape
        output_range = (output.min().item(), output.max().item())
    
    print(f"✅ Forward pass test:")
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Output shape: {output_shape}")
    print(f"   Output range: [{output_range[0]:.3f}, {output_range[1]:.3f}]")
    
    print("\n✅ Syllable LSTM demo completed!") 