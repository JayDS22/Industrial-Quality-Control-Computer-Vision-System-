#!/usr/bin/env python3
"""
Models Package
Industrial Quality Control Computer Vision System
"""

from .yolo_model import YOLODetector, YOLOTrainer
from .resnet_model import ResNetClassifier, ResNetTrainer
from .ensemble import EnsemblePredictor, EnsembleOptimizer
from .tensorrt_optimizer import TensorRTOptimizer

__version__ = "1.0.0"
__author__ = "QC Vision Team"

__all__ = [
    "YOLODetector",
    "YOLOTrainer", 
    "ResNetClassifier",
    "ResNetTrainer",
    "EnsemblePredictor",
    "EnsembleOptimizer",
    "TensorRTOptimizer"
]
