"""
Neural Training Pipeline for Khmer Character-Level LSTM Models

This module provides comprehensive training infrastructure for neural models,
including data preparation, training loops, validation, and model persistence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

from .character_lstm import CharacterLSTMModel, CharacterVocabulary, LSTMConfiguration, ModelTrainingConfig


@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_perplexity: float = 0.0
    val_perplexity: float = 0.0
    learning_rate: float = 0.0
    training_time: float = 0.0
    validation_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_perplexity': self.train_perplexity,
            'val_perplexity': self.val_perplexity,
            'learning_rate': self.learning_rate,
            'training_time': self.training_time,
            'validation_time': self.validation_time
        }


@dataclass
class TrainingResult:
    """Complete training results"""
    best_model_path: str = ""
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    training_history: List[TrainingMetrics] = field(default_factory=list)
    total_training_time: float = 0.0
    model_info: Dict = field(default_factory=dict)
    vocabulary_info: Dict = field(default_factory=dict)
    
    def save_results(self, filepath: str):
        """Save training results to file"""
        results_data = {
            'best_model_path': self.best_model_path,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'total_training_time': self.total_training_time,
            'model_info': self.model_info,
            'vocabulary_info': self.vocabulary_info,
            'training_history': [metrics.to_dict() for metrics in self.training_history]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)


class TrainingDataset(Dataset):
    """
    Dataset for character-level LSTM training
    
    Handles sequence generation, padding, and batching for efficient training
    on Khmer text data.
    """
    
    def __init__(self, 
                 texts: List[str], 
                 vocabulary: CharacterVocabulary,
                 sequence_length: int = 100,
                 stride: int = 50,
                 min_sequence_length: int = 10):
        self.texts = texts
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length
        self.stride = stride
        self.min_sequence_length = min_sequence_length
        
        # Generate sequences
        self.sequences = self._generate_sequences()
        
        self.logger = logging.getLogger("training_dataset")
        self.logger.info(f"Dataset created with {len(self.sequences)} sequences")
    
    def _generate_sequences(self) -> List[Tuple[List[int], List[int]]]:
        """
        Generate input-target sequence pairs for training
        
        Returns:
            List of (input_sequence, target_sequence) pairs
        """
        sequences = []
        
        for text in self.texts:
            if len(text) < self.min_sequence_length:
                continue
            
            # Encode the entire text
            encoded_text = self.vocabulary.encode_text(text, max_length=len(text))
            
            # Generate sliding window sequences
            for i in range(0, len(encoded_text) - self.sequence_length, self.stride):
                input_seq = encoded_text[i:i + self.sequence_length]
                target_seq = encoded_text[i + 1:i + self.sequence_length + 1]
                
                # Ensure sequences are of correct length
                if len(input_seq) == self.sequence_length and len(target_seq) == self.sequence_length:
                    sequences.append((input_seq, target_seq))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.sequences[idx]
        
        # Convert to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        
        return input_tensor, target_tensor
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information"""
        if not self.sequences:
            return {
                'num_sequences': 0,
                'sequence_length': self.sequence_length,
                'stride': self.stride,
                'min_sequence_length': self.min_sequence_length,
                'avg_sequence_length': 0,
                'vocab_coverage': 0,
                'vocab_coverage_ratio': 0,
                'total_characters': 0,
                'error': 'No sequences generated - texts may be too short'
            }
        
        # Calculate statistics
        sequence_lengths = [len(seq[0]) for seq in self.sequences]
        vocab_coverage = set()
        for seq, _ in self.sequences:
            vocab_coverage.update(seq)
        
        return {
            'num_sequences': len(self.sequences),
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'min_sequence_length': self.min_sequence_length,
            'avg_sequence_length': np.mean(sequence_lengths),
            'vocab_coverage': len(vocab_coverage),
            'vocab_coverage_ratio': len(vocab_coverage) / self.vocabulary.vocabulary_size,
            'total_characters': sum(sequence_lengths)
        }


class NeuralTrainer:
    """
    Comprehensive neural model trainer for Khmer character-level LSTM models
    
    Provides complete training infrastructure including:
    - Data preparation and loading
    - Training and validation loops
    - Model checkpointing and persistence
    - Metrics tracking and visualization
    - Early stopping and learning rate scheduling
    """
    
    def __init__(self, 
                 model: CharacterLSTMModel,
                 vocabulary: CharacterVocabulary,
                 config: ModelTrainingConfig):
        self.model = model
        self.vocabulary = vocabulary
        self.config = config
        
        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Training state
        self.training_history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.logger = logging.getLogger("neural_trainer")
        self.logger.info(f"Trainer initialized on device: {self.device}")
    
    def _setup_training_components(self, learning_rate: float):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )
    
    def prepare_data(self, 
                    texts: List[str],
                    sequence_length: int = 100,
                    batch_size: int = 32,
                    validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders
        
        Args:
            texts: List of training texts
            sequence_length: Length of character sequences
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        self.logger.info(f"Preparing data from {len(texts)} texts...")
        
        # Create dataset
        dataset = TrainingDataset(
            texts=texts,
            vocabulary=self.vocabulary,
            sequence_length=sequence_length,
            stride=sequence_length // 2,  # 50% overlap
            min_sequence_length=10
        )
        
        dataset_info = dataset.get_dataset_info()
        self.logger.info(f"Dataset: {dataset_info['num_sequences']} sequences")
        
        # Check if we have any sequences
        if dataset_info['num_sequences'] == 0:
            self.logger.error(f"No training sequences generated! {dataset_info.get('error', '')}")
            self.logger.error(f"Try reducing sequence_length (current: {sequence_length}) or using longer texts")
            raise ValueError(f"No training sequences generated. Current sequence_length: {sequence_length}, "
                           f"but texts may be shorter. Try using sequence_length <= 30 for demo texts.")
        
        self.logger.info(f"Vocab coverage: {dataset_info['vocab_coverage_ratio']:.1%}")
        
        # Split dataset
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.logger.info(f"Data prepared: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train model for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, average_perplexity)
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, (input_seqs, target_seqs) in enumerate(train_loader):
            # Move to device
            input_seqs = input_seqs.to(self.device)
            target_seqs = target_seqs.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Model expects input for next character prediction
            # We use all positions except the last for input
            # and all positions except the first for targets
            input_batch = input_seqs[:, :-1]  # [batch_size, seq_len-1]
            target_batch = target_seqs[:, :-1]  # [batch_size, seq_len-1]
            
            # Forward pass through model
            logits = self.model(input_batch)  # [batch_size, vocab_size]
            
            # Reshape targets for loss calculation
            batch_size = target_batch.size(0)
            
            # For sequence-to-one prediction, we predict the last character
            # Use the last character of each sequence as target
            targets = target_batch[:, -1]  # [batch_size]
            
            # Calculate loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Log progress
            if self.config.verbose and batch_idx % 50 == 0:
                self.logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / total_samples
        avg_perplexity = np.exp(avg_loss)
        
        return avg_loss, avg_perplexity
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, average_perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for input_seqs, target_seqs in val_loader:
                # Move to device
                input_seqs = input_seqs.to(self.device)
                target_seqs = target_seqs.to(self.device)
                
                # Prepare input and targets (same as training)
                input_batch = input_seqs[:, :-1]
                target_batch = target_seqs[:, :-1]
                
                # Forward pass
                logits = self.model(input_batch)
                targets = target_batch[:, -1]
                
                # Calculate loss
                loss = self.criterion(logits, targets)
                
                # Accumulate metrics
                batch_size = target_batch.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_perplexity = np.exp(avg_loss)
        
        return avg_loss, avg_perplexity
    
    def train(self, 
              texts: List[str],
              lstm_config: LSTMConfiguration,
              output_dir: str = "output/neural_training",
              model_name: str = "khmer_char_lstm") -> TrainingResult:
        """
        Complete training pipeline
        
        Args:
            texts: Training texts
            lstm_config: LSTM configuration
            output_dir: Output directory for models and results
            model_name: Base name for saved models
            
        Returns:
            TrainingResult with complete training information
        """
        start_time = time.time()
        
        self.logger.info("🚀 Starting neural model training...")
        self.logger.info(f"Training on {len(texts)} texts")
        self.logger.info(f"Model: {self.model.get_model_info()['total_parameters']:,} parameters")
        
        # Setup training components
        self._setup_training_components(lstm_config.learning_rate)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            texts=texts,
            sequence_length=lstm_config.sequence_length,
            batch_size=lstm_config.batch_size,
            validation_split=self.config.validation_split
        )
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        best_model_path = ""
        
        # Training loop
        for epoch in range(lstm_config.num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_start = time.time()
            train_loss, train_perplexity = self.train_epoch(train_loader)
            train_time = time.time() - train_start
            
            # Validation phase
            val_start = time.time()
            val_loss, val_perplexity = self.validate_epoch(val_loader)
            val_time = time.time() - val_start
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Create metrics
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_perplexity=train_perplexity,
                val_perplexity=val_perplexity,
                learning_rate=current_lr,
                training_time=train_time,
                validation_time=val_time
            )
            
            self.training_history.append(metrics)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch+1}/{lstm_config.num_epochs} ({epoch_time:.1f}s)")
            self.logger.info(f"  Train: Loss={train_loss:.4f}, Perplexity={train_perplexity:.2f}")
            self.logger.info(f"  Val:   Loss={val_loss:.4f}, Perplexity={val_perplexity:.2f}")
            self.logger.info(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if self.config.save_best_only:
                    best_model_path = output_path / f"{model_name}_best_epoch_{epoch+1}.pth"
                    self.model.save_model(str(best_model_path), self.vocabulary)
                    self.logger.info(f"  ✅ Best model saved: {best_model_path}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= lstm_config.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        
        # Create final results
        result = TrainingResult(
            best_model_path=str(best_model_path),
            best_epoch=len(self.training_history) - self.patience_counter,
            best_val_loss=self.best_val_loss,
            training_history=self.training_history,
            total_training_time=total_time,
            model_info=self.model.get_model_info(),
            vocabulary_info=self.vocabulary.get_vocabulary_info()
        )
        
        # Save training results
        results_path = output_path / f"{model_name}_training_results.json"
        result.save_results(str(results_path))
        
        self.logger.info(f"🎯 Training completed in {total_time:.1f}s")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Results saved to: {results_path}")
        
        return result
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.training_history:
            self.logger.warning("No training history to plot")
            return
        
        epochs = [m.epoch for m in self.training_history]
        train_losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history]
        train_perplexities = [m.train_perplexity for m in self.training_history]
        val_perplexities = [m.val_perplexity for m in self.training_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plots
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
        ax1.plot(epochs, val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Perplexity plots
        ax2.plot(epochs, train_perplexities, label='Train Perplexity', color='blue')
        ax2.plot(epochs, val_perplexities, label='Val Perplexity', color='red')
        ax2.set_title('Training and Validation Perplexity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        learning_rates = [m.learning_rate for m in self.training_history]
        ax3.plot(epochs, learning_rates, color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Training time per epoch
        training_times = [m.training_time for m in self.training_history]
        ax4.bar(epochs, training_times, color='orange', alpha=0.7)
        ax4.set_title('Training Time per Epoch')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plot saved to: {save_path}")
        
        plt.show()
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        if not self.training_history:
            return {'error': 'No training history available'}
        
        best_epoch_idx = min(range(len(self.training_history)), 
                           key=lambda i: self.training_history[i].val_loss)
        best_metrics = self.training_history[best_epoch_idx]
        
        final_metrics = self.training_history[-1]
        
        return {
            'total_epochs': len(self.training_history),
            'best_epoch': best_metrics.epoch,
            'best_val_loss': best_metrics.val_loss,
            'best_val_perplexity': best_metrics.val_perplexity,
            'final_train_loss': final_metrics.train_loss,
            'final_val_loss': final_metrics.val_loss,
            'final_train_perplexity': final_metrics.train_perplexity,
            'final_val_perplexity': final_metrics.val_perplexity,
            'total_training_time': sum(m.training_time for m in self.training_history),
            'avg_epoch_time': np.mean([m.training_time + m.validation_time 
                                     for m in self.training_history]),
            'device': str(self.device),
            'early_stopped': self.patience_counter >= self.model.config.patience
        }


if __name__ == "__main__":
    # Demo usage
    print("🧠 NEURAL TRAINER DEMO")
    print("=" * 25)
    
    # Create configurations
    lstm_config = LSTMConfiguration(
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        sequence_length=50,
        batch_size=16,
        num_epochs=5,
        learning_rate=0.001
    )
    
    training_config = ModelTrainingConfig(
        device='cpu',
        verbose=True
    )
    
    print(f"LSTM Config: {lstm_config.embedding_dim}D embeddings, {lstm_config.hidden_dim}D hidden")
    print(f"Training Config: {training_config.device} device")
    
    # Sample training texts
    sample_texts = [
        "នេះជាការសាកល្បងអត្ថបទខ្មែរដ៏វែងសម្រាប់ការបណ្តុះបណ្តាលម៉ូដែល។",
        "ភាសាខ្មែរមានលក្ខណៈពិសេសនិងស្រស់ស្អាតណាស់។", 
        "ការអប់រំជាមូលដ្ឋានសំខាន់សម្រាប់ការអភិវឌ្ឍន៍ប្រទេសជាតិ។",
        "វប្បធម៌ខ្មែរមានប្រវត្តិដ៏យូរលង់និងសម្បូរបែប។",
        "កម្ពុជាជាប្រទេសដែលមានធនធានធម្មជាតិច្រើនប្រភេទ។"
    ]
    
    print(f"Sample texts: {len(sample_texts)} training samples")
    
    # Build vocabulary
    vocabulary = CharacterVocabulary(lstm_config)
    vocab_stats = vocabulary.build_vocabulary(sample_texts)
    
    print(f"✅ Vocabulary: {vocab_stats['vocabulary_size']} characters")
    print(f"   Coverage: {vocab_stats['coverage']:.1%}")
    
    # Create model
    model = CharacterLSTMModel(lstm_config, vocabulary.vocabulary_size)
    
    # Create trainer
    trainer = NeuralTrainer(model, vocabulary, training_config)
    
    print(f"✅ Trainer created on device: {trainer.device}")
    
    # Test data preparation
    train_loader, val_loader = trainer.prepare_data(
        texts=sample_texts,
        sequence_length=lstm_config.sequence_length,
        batch_size=lstm_config.batch_size
    )
    
    print(f"✅ Data prepared: {len(train_loader)} train, {len(val_loader)} val batches")
    
    # Test single epoch training
    print("\n🔄 Testing single epoch training...")
    train_loss, train_perplexity = trainer.train_epoch(train_loader)
    val_loss, val_perplexity = trainer.validate_epoch(val_loader)
    
    print(f"   Train: Loss={train_loss:.4f}, Perplexity={train_perplexity:.2f}")
    print(f"   Val:   Loss={val_loss:.4f}, Perplexity={val_perplexity:.2f}")
    
    print("\n✅ Neural trainer demo completed!") 