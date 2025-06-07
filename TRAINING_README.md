# Khmer Spellchecker Model Training System

This comprehensive training system trains all models required for the production Khmer spellchecker, including statistical models, neural models, and ensemble configurations.

## 🎯 Overview

The training pipeline creates:
- **Character N-gram models** (3-gram, 4-gram, 5-gram) for character-level error detection
- **Syllable N-gram models** (2-gram, 3-gram, 4-gram) for syllable-level validation  
- **Syllable-level LSTM neural models** for advanced sequence modeling
- **Rule-based validation models** for linguistic validation
- **Ensemble integration configuration** for production deployment

## 🚀 Quick Start

### 1. Basic Training

Train all models with default configuration:

```bash
python train_models.py --data_dir /path/to/your/khmer/data
```

### 2. Quick Demo Training

Train with sample data for testing:

```bash
python demo_training.py
```

### 3. Production Training

Train with optimized production configuration:

```bash
python train_models.py --data_dir /path/to/data --config production_training_config.json
```

## 📁 Data Requirements

### Input Data Format
- **Text files** containing Khmer content (UTF-8 encoding)
- **Mixed content** supported (Khmer + English + numbers)
- **Minimum**: 1MB of text data recommended for meaningful models
- **Optimal**: 100MB+ for production-quality models

### Data Directory Structure
```
data/
├── file1.txt
├── file2.txt
└── file3.txt
```

### Content Quality
- **Khmer ratio**: Minimum 60% Khmer characters (configurable)
- **Text length**: 50-10,000 characters per paragraph (configurable)
- **Encoding**: UTF-8 required
- **Format**: Plain text, HTML, or mixed content

## ⚙️ Configuration Options

### Command Line Arguments

```bash
python train_models.py [OPTIONS]

Required:
  --data_dir PATH         Directory containing training data

Optional:
  --output_dir PATH       Output directory (default: output/models)
  --config PATH           Configuration file (JSON)
  --no_neural            Disable neural model training
  --max_files N          Maximum files to process
  --quick                Quick training (reduced parameters)
```

### Configuration File

Create a JSON configuration file to customize training parameters:

```json
{
  "character_ngrams": [3, 4, 5],
  "syllable_ngrams": [2, 3, 4],
  "neural_enabled": true,
  "neural_epochs": 10,
  "min_khmer_ratio": 0.6,
  "validate_models": true
}
```

See `production_training_config.json` for complete configuration options.

### Key Configuration Parameters

#### Data Processing
- `max_files`: Maximum files to process (null = all files)
- `min_khmer_ratio`: Minimum Khmer character ratio (0.6 = 60%)
- `min_text_length`: Minimum paragraph length (50 characters)
- `max_text_length`: Maximum paragraph length (10,000 characters)
- `batch_size`: Processing batch size (1000)

#### Statistical Models
- `character_ngrams`: N-gram sizes for character models [3, 4, 5]
- `syllable_ngrams`: N-gram sizes for syllable models [2, 3, 4]
- `char_smoothing_method`: Smoothing technique ("good_turing", "laplace", "simple_backoff")
- `char_filter_non_khmer`: Filter non-Khmer characters (true)

#### Neural Models
- `neural_enabled`: Enable neural model training (true)
- `neural_sequence_length`: Sequence length for LSTM (20)
- `neural_vocab_size`: Maximum vocabulary size (8000)
- `neural_epochs`: Training epochs (10)
- `neural_batch_size`: Training batch size (32)

#### Ensemble Configuration
- `ensemble_neural_weight`: Neural model weight (0.35)
- `ensemble_statistical_weight`: Statistical model weight (0.45)
- `ensemble_rule_weight`: Rule-based model weight (0.20)

## 📊 Output Structure

After training, the following directory structure is created:

```
output/models/
├── statistical_models/
│   ├── character_3gram_model.json
│   ├── character_3gram_model.pkl
│   ├── character_4gram_model.json
│   ├── character_4gram_model.pkl
│   ├── character_5gram_model.json
│   ├── character_5gram_model.pkl
│   ├── syllable_2gram_model.json
│   ├── syllable_2gram_model.pkl
│   ├── syllable_3gram_model.json
│   ├── syllable_3gram_model.pkl
│   ├── syllable_4gram_model.json
│   └── syllable_4gram_model.pkl
├── neural_models/
│   ├── syllable_lstm_model.pth
│   ├── syllable_vocabulary.json
│   └── training_history.json
├── ensemble_configs/
│   ├── production_ensemble_config.json
│   └── api_config.json
├── training_report.json
└── training_summary.txt
```

### Model Files

#### Statistical Models
- **JSON format**: Human-readable, for analysis and debugging
- **Pickle format**: Binary format, fast loading for production

#### Neural Models
- **PyTorch format**: `.pth` files containing trained neural networks
- **Vocabulary**: JSON files with syllable-to-index mappings
- **History**: Training metrics and loss curves

#### Configuration Files
- **Production config**: Complete ensemble configuration
- **API config**: Simplified configuration for API deployment

## 🎯 Training Scenarios

### 1. Development Training
For development and testing:
```bash
python train_models.py --data_dir sample_data --quick --no_neural
```

### 2. Research Training
For research with full statistical models:
```bash
python train_models.py --data_dir /path/to/data --config research_config.json
```

### 3. Production Training
For production deployment:
```bash
python train_models.py --data_dir /path/to/large/corpus --config production_training_config.json
```

### 4. Incremental Training
Continue from existing models:
```bash
python train_models.py --data_dir /path/to/new/data --output_dir output/models
```

## 🔧 Performance Optimization

### Memory Usage
- **Batch processing**: Process texts in configurable batches
- **Model size control**: Limit vocabulary sizes for memory efficiency
- **Incremental training**: Process files one at a time

### Speed Optimization
- **Parallel processing**: Syllable segmentation uses optimized methods
- **Model caching**: Reuse loaded models during validation
- **Quick training**: Reduced parameters for faster testing

### Storage Optimization
- **Dual formats**: JSON for debugging, Pickle for production
- **Compression**: Models use efficient serialization
- **Selective saving**: Configure which intermediate files to save

## 📈 Training Progress and Monitoring

### Real-time Progress
The training system provides comprehensive progress monitoring:

```
2025-01-07 10:30:15 - INFO - STEP 1: Loading and Preprocessing Data
2025-01-07 10:30:16 - INFO - Loading files from /path/to/data
2025-01-07 10:30:17 - INFO - Loaded 4 documents
2025-01-07 10:30:18 - INFO - Processing document: file1.txt (175.8MB)
2025-01-07 10:30:20 - INFO -   - Processed 3339 paragraphs, kept 2941 valid texts
```

### Training Reports
After completion, detailed reports are generated:

- **JSON report**: Machine-readable training statistics
- **Text summary**: Human-readable training summary
- **Model validation**: Validation results for all trained models

### Error Handling
- **Graceful degradation**: Continue training if some components fail
- **Error tracking**: Comprehensive error logging and reporting
- **Recovery**: Ability to resume from intermediate states

## 🧪 Testing and Validation

### Automatic Validation
All trained models are automatically validated with test texts:

```python
test_texts = [
    "នេះជាការសាកល្បងអត្ថបទខ្មែរ។",
    "ការអប់រំជាមូលដ្ឋានសំខាន់។",
    "កំហុស test error ៕៖។"  # Mixed content with errors
]
```

### Validation Metrics
- **Model loading**: Verify all models load correctly
- **Perplexity calculation**: Test statistical model functionality
- **Neural prediction**: Verify neural model inference
- **Integration testing**: Test ensemble configuration

### Performance Benchmarks
- **Processing speed**: Texts per second
- **Memory usage**: Peak memory consumption
- **Model accuracy**: Validation perplexity scores
- **Coverage**: Vocabulary coverage statistics

## 🐛 Troubleshooting

### Common Issues

#### 1. Data Loading Errors
```
Error: No documents found in /path/to/data
```
**Solution**: Ensure data directory contains text files and is accessible.

#### 2. Memory Errors
```
Error: Out of memory during neural training
```
**Solution**: Reduce `neural_batch_size`, `neural_vocab_size`, or use `--no_neural`.

#### 3. Encoding Errors
```
Error: UnicodeDecodeError
```
**Solution**: Ensure all text files are UTF-8 encoded.

#### 4. Model Validation Failures
```
Warning: Some models failed validation
```
**Solution**: Check training report for specific errors and model paths.

### Debug Mode
Enable detailed logging for troubleshooting:

```json
{
  "detailed_logging": true,
  "validate_models": true
}
```

### Performance Issues
For slow training:
- Use `--quick` flag for reduced parameters
- Reduce `neural_epochs` and `neural_vocab_size`
- Limit data with `--max_files`
- Disable neural training with `--no_neural`

## 🔄 Integration with Production

### Using Trained Models

After training, models can be used in the production spellchecker:

```python
from production.khmer_spellchecker_api import KhmerSpellcheckerService

# Initialize with trained models
service = KhmerSpellcheckerService("output/models/ensemble_configs/api_config.json")

# Validate text
result = service.validate_text("នេះជាការសាកល្បង")
print(f"Valid: {result.is_valid}")
```

### Model Updates
To update models with new data:
1. Add new text files to data directory
2. Run training with same output directory
3. Models will be updated with new data
4. Restart production service to load updated models

### Performance Monitoring
Monitor model performance in production:
- Track validation accuracy over time
- Monitor processing speed and memory usage
- Collect user feedback for model improvement

## 📚 Additional Resources

### Configuration Examples
- `production_training_config.json`: Production-ready configuration
- `demo_training_config.json`: Quick demo configuration (generated by demo)

### Demo Scripts
- `demo_training.py`: Interactive training demonstrations
- `train_models.py`: Main training script
- `demo_phase_*.py`: Individual component demonstrations

### Documentation
- `docs/changes.md`: Detailed development history
- `TRAINING_README.md`: This comprehensive guide
- Individual module docstrings for technical details

## 🎉 Success Metrics

### Training Success Indicators
- ✅ All models trained without errors
- ✅ Model validation passes (100% success rate)
- ✅ Training report generated successfully
- ✅ Output files created in correct structure
- ✅ Statistical models show reasonable perplexity scores
- ✅ Neural models converge during training

### Production Readiness
- Models load correctly in production environment
- Validation speed meets performance requirements (<100ms)
- Memory usage within acceptable limits (<1GB)
- Ensemble configuration optimized for accuracy

Your Khmer spellchecker models are now ready for production deployment! 🚀 