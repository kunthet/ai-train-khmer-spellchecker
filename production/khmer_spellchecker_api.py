"""
Phase 3.5: Production Khmer Spellchecker API

This module provides a production-ready FastAPI service that integrates neural-statistical
models with ensemble optimization for comprehensive Khmer spellchecking capabilities.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import time
import logging
import asyncio
import json
from pathlib import Path
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import our spellchecker components
from neural_models.neural_statistical_integration import (
    NeuralStatisticalIntegrator, 
    IntegrationConfiguration,
    IntegrationResult
)
from neural_models.hybrid_ensemble_optimizer import (
    HybridEnsembleOptimizer,
    ParameterSpace,
    OptimizationObjective
)


# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/khmer_spellchecker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("khmer_spellchecker_api")

# Pydantic models for API
class TextInput(BaseModel):
    """Input model for text validation"""
    text: str = Field(..., min_length=1, max_length=10000, description="Khmer text to validate")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Validation options")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v

class BatchTextInput(BaseModel):
    """Input model for batch text validation"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to validate")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Validation options")
    
    @validator('texts')
    def validate_texts(cls, v):
        for text in v:
            if not text.strip():
                raise ValueError("All texts must be non-empty")
        return v

class ValidationError(BaseModel):
    """Validation error model"""
    position: int = Field(..., description="Error position in text")
    syllable: str = Field(..., description="Problematic syllable")
    error_type: str = Field(..., description="Type of error detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Error confidence score")
    suggestions: List[str] = Field(default=[], description="Correction suggestions")
    sources: List[str] = Field(default=[], description="Error detection sources")

class ValidationResponse(BaseModel):
    """Response model for text validation"""
    request_id: str = Field(..., description="Unique request identifier")
    text: str = Field(..., description="Original input text")
    is_valid: bool = Field(..., description="Whether text is valid")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall validation confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    errors: List[ValidationError] = Field(default=[], description="Detected errors")
    statistics: Dict[str, Any] = Field(default={}, description="Processing statistics")

class BatchValidationResponse(BaseModel):
    """Response model for batch validation"""
    request_id: str = Field(..., description="Unique request identifier")
    total_texts: int = Field(..., description="Total number of texts processed")
    valid_texts: int = Field(..., description="Number of valid texts")
    total_errors: int = Field(..., description="Total errors detected")
    processing_time: float = Field(..., description="Total processing time")
    results: List[ValidationResponse] = Field(..., description="Individual validation results")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    uptime: float = Field(..., description="Service uptime in seconds")
    processed_requests: int = Field(..., description="Total processed requests")

class MetricsResponse(BaseModel):
    """Metrics response model"""
    uptime: float = Field(..., description="Service uptime in seconds")
    total_requests: int = Field(..., description="Total requests processed")
    total_texts: int = Field(..., description="Total texts validated")
    total_errors: int = Field(..., description="Total errors detected")
    average_processing_time: float = Field(..., description="Average processing time per text")
    throughput: float = Field(..., description="Texts processed per second")
    model_performance: Dict[str, Any] = Field(..., description="Model performance metrics")


class KhmerSpellcheckerService:
    """
    Production Khmer spellchecker service with comprehensive error handling,
    monitoring, and performance optimization.
    """
    
    def __init__(self, config_path: str = "production/config.json"):
        self.config_path = config_path
        self.start_time = time.time()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize spellchecker components
        self.integrator: Optional[NeuralStatisticalIntegrator] = None
        self.optimizer: Optional[HybridEnsembleOptimizer] = None
        
        # Performance tracking
        self.metrics = {
            'total_requests': 0,
            'total_texts': 0,
            'total_errors': 0,
            'total_processing_time': 0.0,
            'model_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Simple cache for frequently validated texts
        self.cache: Dict[str, ValidationResponse] = {}
        self.cache_max_size = self.config.get('cache_max_size', 1000)
        
        logger.info("Khmer Spellchecker Service initialized")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load service configuration"""
        default_config = {
            "neural_weight": 0.4,
            "statistical_weight": 0.4,
            "rule_weight": 0.2,
            "consensus_threshold": 0.6,
            "error_confidence_threshold": 0.5,
            "max_text_length": 10000,
            "batch_size": 32,
            "enable_caching": True,
            "cache_max_size": 1000,
            "model_paths": {
                "neural_model": "output/neural_models/syllable_lstm_model.pth",
                "statistical_models": "output/statistical_models"
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info("Using default configuration")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}, using defaults")
        
        return default_config
    
    async def initialize_models(self):
        """Initialize spellchecker models asynchronously"""
        try:
            logger.info("Initializing spellchecker models...")
            
            # Create integration configuration
            integration_config = IntegrationConfiguration(
                neural_weight=self.config['neural_weight'],
                statistical_weight=self.config['statistical_weight'],
                rule_weight=self.config['rule_weight'],
                consensus_threshold=self.config['consensus_threshold'],
                error_confidence_threshold=self.config['error_confidence_threshold'],
                batch_size=self.config['batch_size']
            )
            
            # Initialize neural-statistical integrator
            self.integrator = NeuralStatisticalIntegrator(integration_config)
            
            # Try to load existing models
            models_loaded = await self._load_models()
            
            if not models_loaded:
                # Create minimal models for demonstration
                logger.info("Creating minimal models for production demo...")
                demo_texts = [
                    "នេះជាការសាកល្បងអត្ថបទដ៏ល្អមួយ។",
                    "ការអប់រំជាមូលដ្ឋានសំខាន់សម្រាប់ការអភិវឌ្ឍន៍។",
                    "វប្បធម៌ខ្មែរមានប្រវត្តិដ៏យូរលង់។",
                    "កម្ពុជាជាប្រទេសដែលមានធនធានធម្មជាតិច្រើនប្រភេទ។"
                ] * 50  # Replicate for better training
                
                success = self.integrator.create_minimal_models(demo_texts)
                if not success:
                    raise Exception("Failed to create minimal models")
            
            # Initialize optimizer
            self.optimizer = HybridEnsembleOptimizer()
            
            self.metrics['model_loads'] += 1
            logger.info("✅ Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def _load_models(self) -> bool:
        """Try to load existing trained models"""
        try:
            models_loaded = 0
            
            # Try to load neural model
            neural_paths = [
                self.config['model_paths']['neural_model'],
                "output/neural_models/syllable_lstm_model.pth",
                "models/syllable_lstm_model.pth"
            ]
            
            for path in neural_paths:
                if self.integrator.load_neural_model(path):
                    models_loaded += 1
                    break
            
            # Try to load statistical models
            statistical_dirs = [
                self.config['model_paths']['statistical_models'],
                "output/statistical_models",
                "models/statistical"
            ]
            
            for directory in statistical_dirs:
                count = self.integrator.load_statistical_models(directory)
                if count > 0:
                    models_loaded += count
                    break
            
            # Load rule validator
            if self.integrator.load_rule_validator():
                models_loaded += 1
            
            return models_loaded > 0
            
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
            return False
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return str(uuid.uuid4())[:8]
    
    def _check_cache(self, text: str) -> Optional[ValidationResponse]:
        """Check cache for previously validated text"""
        if not self.config.get('enable_caching', True):
            return None
        
        text_hash = str(hash(text))
        if text_hash in self.cache:
            self.metrics['cache_hits'] += 1
            cached_result = self.cache[text_hash]
            # Update request ID for new request
            cached_result.request_id = self._generate_request_id()
            return cached_result
        
        self.metrics['cache_misses'] += 1
        return None
    
    def _cache_result(self, text: str, result: ValidationResponse):
        """Cache validation result"""
        if not self.config.get('enable_caching', True):
            return
        
        # Implement simple LRU by clearing cache when full
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest 25% of entries
            items_to_remove = list(self.cache.keys())[:len(self.cache) // 4]
            for key in items_to_remove:
                del self.cache[key]
        
        text_hash = str(hash(text))
        self.cache[text_hash] = result
    
    async def validate_text(self, text: str, options: Dict[str, Any] = None) -> ValidationResponse:
        """Validate a single text"""
        request_id = self._generate_request_id()
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self._check_cache(text)
            if cached_result:
                logger.info(f"Request {request_id}: Cache hit for text validation")
                return cached_result
            
            # Validate input
            if len(text) > self.config['max_text_length']:
                raise ValueError(f"Text too long (max {self.config['max_text_length']} characters)")
            
            # Perform validation
            result = self.integrator.validate_text(text)
            
            # Convert to API response format
            errors = [
                ValidationError(
                    position=error.position,
                    syllable=error.syllable,
                    error_type=error.error_type,
                    confidence=error.confidence,
                    suggestions=error.suggestions,
                    sources=error.sources
                )
                for error in result.errors
            ]
            
            processing_time = time.time() - start_time
            
            response = ValidationResponse(
                request_id=request_id,
                text=text,
                is_valid=result.is_valid,
                overall_confidence=result.overall_confidence,
                processing_time=processing_time,
                errors=errors,
                statistics={
                    'syllables_count': len(result.syllables),
                    'method_agreement': result.method_agreement,
                    'neural_perplexity': result.neural_perplexity,
                    'statistical_entropy': result.statistical_entropy,
                    'rule_based_score': result.rule_based_score
                }
            )
            
            # Cache the result
            self._cache_result(text, response)
            
            # Update metrics
            self.metrics['total_requests'] += 1
            self.metrics['total_texts'] += 1
            self.metrics['total_errors'] += len(errors)
            self.metrics['total_processing_time'] += processing_time
            
            logger.info(f"Request {request_id}: Validated text ({len(text)} chars) in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Request {request_id}: Validation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
    
    async def validate_batch(self, texts: List[str], options: Dict[str, Any] = None) -> BatchValidationResponse:
        """Validate multiple texts in batch"""
        request_id = self._generate_request_id()
        start_time = time.time()
        
        try:
            # Validate all texts
            results = []
            for text in texts:
                result = await self.validate_text(text, options)
                results.append(result)
            
            # Calculate batch statistics
            valid_texts = sum(1 for r in results if r.is_valid)
            total_errors = sum(len(r.errors) for r in results)
            processing_time = time.time() - start_time
            
            response = BatchValidationResponse(
                request_id=request_id,
                total_texts=len(texts),
                valid_texts=valid_texts,
                total_errors=total_errors,
                processing_time=processing_time,
                results=results
            )
            
            logger.info(f"Batch {request_id}: Validated {len(texts)} texts in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Batch {request_id}: Validation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")
    
    def get_health(self) -> HealthResponse:
        """Get service health status"""
        uptime = time.time() - self.start_time
        
        models_status = {
            'neural_model': self.integrator.neural_model is not None if self.integrator else False,
            'statistical_models': len(self.integrator.statistical_models) > 0 if self.integrator else False,
            'rule_validator': self.integrator.rule_validator is not None if self.integrator else False,
            'optimizer': self.optimizer is not None
        }
        
        return HealthResponse(
            status="healthy" if all(models_status.values()) else "degraded",
            timestamp=datetime.now(),
            version="3.5.0",
            models_loaded=models_status,
            uptime=uptime,
            processed_requests=self.metrics['total_requests']
        )
    
    def get_metrics(self) -> MetricsResponse:
        """Get service metrics"""
        uptime = time.time() - self.start_time
        
        avg_processing_time = (
            self.metrics['total_processing_time'] / self.metrics['total_texts']
            if self.metrics['total_texts'] > 0 else 0.0
        )
        
        throughput = self.metrics['total_texts'] / uptime if uptime > 0 else 0.0
        
        return MetricsResponse(
            uptime=uptime,
            total_requests=self.metrics['total_requests'],
            total_texts=self.metrics['total_texts'],
            total_errors=self.metrics['total_errors'],
            average_processing_time=avg_processing_time,
            throughput=throughput,
            model_performance={
                'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0.0,
                'error_rate': self.metrics['total_errors'] / self.metrics['total_texts']
                if self.metrics['total_texts'] > 0 else 0.0,
                'model_loads': self.metrics['model_loads']
            }
        )


# Global service instance
service = KhmerSpellcheckerService()

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Khmer Spellchecker API...")
    await service.initialize_models()
    logger.info("✅ Khmer Spellchecker API started successfully")
    yield
    # Shutdown
    logger.info("Shutting down Khmer Spellchecker API...")

# Create FastAPI app
app = FastAPI(
    title="Khmer Spellchecker API",
    description="Production-ready Khmer spellchecker with neural-statistical integration",
    version="3.5.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# API Endpoints
@app.post("/validate", response_model=ValidationResponse)
async def validate_text(input_data: TextInput):
    """Validate a single Khmer text"""
    return await service.validate_text(input_data.text, input_data.options)

@app.post("/validate/batch", response_model=BatchValidationResponse)
async def validate_batch(input_data: BatchTextInput):
    """Validate multiple Khmer texts in batch"""
    return await service.validate_batch(input_data.texts, input_data.options)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Get service health status"""
    return service.get_health()

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get service performance metrics"""
    return service.get_metrics()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Khmer Spellchecker API",
        "version": "3.5.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    uvicorn.run(
        "production.khmer_spellchecker_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    ) 