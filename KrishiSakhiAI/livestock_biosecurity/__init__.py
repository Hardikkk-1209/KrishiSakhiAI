"""
🐄 Livestock Biosecurity ML Module
===================================
AI-powered livestock health monitoring and biosecurity system.

Components:
- DataGenerator: Synthetic IoT sensor data for training
- HealthPredictor: Multi-model health risk classification
- AnomalyDetector: Isolation Forest anomaly detection
- GaitAnalyzer: Computer vision-based gait/lameness scoring
- AlertSystem: Biosecurity alert generation & management
"""

from .data_generator import LivestockDataGenerator
from .models import HealthPredictor, AnomalyDetector, GaitPredictor, DiseaseForecaster
from .cv_module import GaitAnalyzer, BehaviorAnalyzer
from .alert_system import BiosecurityAlertSystem

__version__ = "1.0.0"
__all__ = [
    'LivestockDataGenerator',
    'HealthPredictor',
    'AnomalyDetector', 
    'GaitPredictor',
    'DiseaseForecaster',
    'GaitAnalyzer',
    'BehaviorAnalyzer',
    'BiosecurityAlertSystem',
]
