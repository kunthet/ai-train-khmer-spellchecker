"""
Production Package for Khmer Spellchecker

This package provides production-ready components for deploying the Khmer spellchecker
including FastAPI service, configuration management, and deployment infrastructure.
"""

from .khmer_spellchecker_api import (
    KhmerSpellcheckerService,
    app,
    TextInput,
    BatchTextInput,
    ValidationError,
    ValidationResponse,
    BatchValidationResponse,
    HealthResponse,
    MetricsResponse
)

__all__ = [
    'KhmerSpellcheckerService',
    'app',
    'TextInput',
    'BatchTextInput', 
    'ValidationError',
    'ValidationResponse',
    'BatchValidationResponse',
    'HealthResponse',
    'MetricsResponse'
]

__version__ = "3.5.0" 