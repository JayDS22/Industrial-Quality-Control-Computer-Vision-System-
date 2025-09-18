#!/usr/bin/env python3
"""
ResNet-50 Classification Model for Quality Control
Handles defect classification and severity assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time
from PIL import Image

logger = logging.getLogger(__name__)

class ResNetClassifier:
    """ResNet-50 based defect classification system"""
    
    def __init__(self, 
                 model_path: str = "models/resnet50_qc.pt",
                 num_classes: int = 5,
                 device: str = "auto"):
        """
        Initialize ResNet classifier
        
        Args:
            model_path: Path to trained ResNet model
            num_classes: Number of defect classes
            device: Device to run inference on
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self.class_names = [
            "crack", "scratch", "dent", "discoloration", "contamination"
        ]
        self.severity_mapping = {
            0: "minor", 1: "major", 2: "critical"
        }
        
        self._setup_transforms()
        self.load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device for inference"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_model(self) -> nn.Module:
        """Build ResNet-50 model architecture"""
        # Load pre-trained ResNet-50
        model = models.resnet50(pretrained=True)
        
        # Modify final layer for our classes
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        return model
    
    def load_model(self) -> bool:
        """Load trained ResNet model"""
        try:
            # Build model architecture
            self.model = self._build_model()
            
            # Try to load trained weights
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                    logger.info(f"Loaded model checkpoint from {self.model_path}")
                else:
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"Loaded model weights from {self.model_path}")
            except FileNotFoundError:
                logger.warning(f"Model file {self.model_path} not found. Using pre-trained ResNet-50")
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"ResNet classifier loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ResNet model: {str(e)}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for ResNet inference"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 1:
                # Decode if it's encoded
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run ResNet classification on image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing classification results
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {"error": "Model not loaded"}
        
        try:
            start_time = time.time()
            
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            inference_time = time.time() - start_time
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
            
            # Create results
            result = {
                "predicted_class": self.class_names[predicted_class.item()],
                "confidence": float(confidence.item()),
                "class_probabilities": {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(all_probs)
                },
                "severity": self._predict_severity(predicted_class.item(), confidence.item()),
                "inference_time_ms": inference_time * 1000
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during ResNet prediction: {str(e)}")
            return {"error": str(e)}
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Run batch classification on multiple images"""
        if self.model is None:
            return [{"error": "Model not loaded"} for _ in images]
        
        try:
            start_time = time.time()
            
            # Preprocess all images
            image_tensors = []
            for img in images:
                tensor = self.preprocess_image(img)
                image_tensors.append(tensor)
            
            # Stack tensors
            batch_tensor = torch.cat(image_tensors, dim=0)
            
            # Run batch inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
            
            total_time = time.time() - start_time
            
            # Parse results
            results = []
            for i in range(len(images)):
                all_probs = probabilities[i].cpu().numpy()
                
                result = {
                    "predicted_class": self.class_names[predicted_classes[i].item()],
                    "confidence": float(confidences[i].item()),
                    "class_probabilities": {
                        self.class_names[j]: float(prob) 
                        for j, prob in enumerate(all_probs)
                    },
                    "severity": self._predict_severity(
                        predicted_classes[i].item(), 
                        confidences[i].item()
                    ),
                    "batch_index": i
                }
                results.append(result)
            
            # Add batch timing info
            for result in results:
                result["batch_inference_time_ms"] = total_time * 1000
                result["avg_time_per_image_ms"] = (total_time * 1000) / len(images)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            return [{"error": str(e)} for _ in images]
    
    def _predict_severity(self, class_id: int, confidence: float) -> str:
        """Predict defect severity based on class and confidence"""
        # Define severity rules
        critical_classes = [0, 2]  # crack, dent
        major_classes = [1, 3]     # scratch, discoloration
        minor_classes = [4]        # contamination
        
        if class_id in critical_classes and confidence > 0.8:
            return "critical"
        elif class_id in critical_classes and confidence > 0.6:
            return "major"
        elif class_id in major_classes and confidence > 0.8:
            return "major"
        elif class_id in major_classes and confidence > 0.6:
            return "minor"
        elif class_id in minor_classes:
            return "minor"
        else:
            return "minor"
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract feature vector from image"""
        if self.model is None:
            return None
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Extract features from second-to-last layer
            with torch.no_grad():
                features = self.model.features(image_tensor)
                features = self.model.avgpool(features)
                features = torch.flatten(features, 1)
            
            return features.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "model_loaded": self.model is not None,
            "input_size": (224, 224)
        }


class ResNetTrainer:
    """Training utilities for ResNet model"""
    
    def __init__(self, num_classes: int = 5, device: str = "auto"):
        """Initialize ResNet trainer"""
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
    
    def build_model(self) -> nn.Module:
        """Build ResNet model for training"""
        model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers
        for param in model.layer4.parameters():
            param.requires_grad = True
        
        # Replace classifier
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        return model.to(self.device)
    
    def setup_training(self, learning_rate: float = 0.001):
        """Setup training components"""
        self.model = self.build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )
        
        logger.info("Training setup completed")
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = running_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def save_model(self, path: str, epoch: int, accuracy: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'num_classes': self.num_classes
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
