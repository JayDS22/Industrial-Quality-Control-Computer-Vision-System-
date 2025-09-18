#!/usr/bin/env python3
"""
YOLOv8 Object Detection Model for Quality Control
Handles defect detection and localization
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Dict, Tuple, Optional
import time

logger = logging.getLogger(__name__)

class YOLODetector:
    """YOLOv8-based defect detection system"""
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.7,
                 nms_threshold: float = 0.5,
                 device: str = "auto"):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = self._get_device(device)
        self.model = None
        self.class_names = []
        self.load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device for inference"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self) -> bool:
        """Load YOLO model"""
        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to device
            if self.device == "cuda":
                self.model.to("cuda")
            
            # Get class names
            self.class_names = list(self.model.names.values()) if self.model.names else [
                "crack", "scratch", "dent", "discoloration", "contamination"
            ]
            
            logger.info(f"YOLO model loaded successfully on {self.device}")
            logger.info(f"Detected classes: {self.class_names}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO inference"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 1:
                # If it's a 1D array (encoded image), decode it
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run YOLO prediction on image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {"error": "Model not loaded"}
        
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run inference
            results = self.model(
                processed_image,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            
            # Parse results
            detections = self._parse_results(results[0], processed_image.shape)
            
            return {
                "detections": detections,
                "inference_time_ms": inference_time * 1000,
                "image_shape": processed_image.shape[:2],
                "total_detections": len(detections)
            }
            
        except Exception as e:
            logger.error(f"Error during YOLO prediction: {str(e)}")
            return {"error": str(e)}
    
    def _parse_results(self, results, image_shape: Tuple) -> List[Dict]:
        """Parse YOLO results into structured format"""
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                
                detection = {
                    "id": i,
                    "class": self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                    "confidence": float(conf),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "center_x": (x1 + x2) / 2,
                        "center_y": (y1 + y2) / 2
                    },
                    "area": (x2 - x1) * (y2 - y1),
                    "severity": self._classify_severity(conf, (x2 - x1) * (y2 - y1))
                }
                
                detections.append(detection)
        
        return detections
    
    def _classify_severity(self, confidence: float, area: float) -> str:
        """Classify defect severity based on confidence and area"""
        # Normalize area (assuming max image size of 1024x1024)
        normalized_area = area / (1024 * 1024)
        
        if confidence > 0.9 or normalized_area > 0.1:
            return "critical"
        elif confidence > 0.8 or normalized_area > 0.05:
            return "major"
        else:
            return "minor"
    
    def batch_predict(self, images: List[np.ndarray]) -> List[Dict]:
        """Run batch prediction on multiple images"""
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.predict(image)
            result["batch_index"] = i
            results.append(result)
        
        return results
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Visualize detections on image"""
        vis_image = image.copy()
        
        # Color map for different classes
        colors = {
            "crack": (255, 0, 0),      # Red
            "scratch": (0, 255, 0),    # Green
            "dent": (0, 0, 255),       # Blue
            "discoloration": (255, 255, 0),  # Yellow
            "contamination": (255, 0, 255)   # Magenta
        }
        
        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            severity = detection["severity"]
            
            # Get color for class
            color = colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (bbox["x1"], bbox["y1"]),
                (bbox["x2"], bbox["y2"]),
                color,
                2
            )
            
            # Draw label
            label = f"{class_name}: {confidence:.2f} ({severity})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(
                vis_image,
                (bbox["x1"], bbox["y1"] - label_size[1] - 10),
                (bbox["x1"] + label_size[0], bbox["y1"]),
                color,
                -1
            )
            
            cv2.putText(
                vis_image,
                label,
                (bbox["x1"], bbox["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return vis_image
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "class_names": self.class_names,
            "model_loaded": self.model is not None
        }
    
    def update_thresholds(self, confidence: Optional[float] = None, nms: Optional[float] = None):
        """Update detection thresholds"""
        if confidence is not None:
            self.confidence_threshold = confidence
            logger.info(f"Updated confidence threshold to {confidence}")
        
        if nms is not None:
            self.nms_threshold = nms
            logger.info(f"Updated NMS threshold to {nms}")


class YOLOTrainer:
    """Training utilities for YOLO model"""
    
    def __init__(self, model_size: str = "n"):
        """
        Initialize YOLO trainer
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        """
        self.model_size = model_size
        self.model = None
    
    def prepare_training(self, data_config: str):
        """Prepare model for training"""
        try:
            self.model = YOLO(f"yolov8{self.model_size}.pt")
            logger.info(f"YOLO{self.model_size} model prepared for training")
            return True
        except Exception as e:
            logger.error(f"Error preparing training: {str(e)}")
            return False
    
    def train(self, 
              data_config: str,
              epochs: int = 100,
              batch_size: int = 16,
              img_size: int = 640,
              device: str = "auto"):
        """Train YOLO model"""
        if self.model is None:
            logger.error("Model not prepared for training")
            return None
        
        try:
            results = self.model.train(
                data=data_config,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device=device,
                save=True,
                save_period=10,
                cache=True,
                augment=True
            )
            
            logger.info("Training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return None
    
    def validate(self, data_config: str):
        """Validate trained model"""
        if self.model is None:
            logger.error("No model to validate")
            return None
        
        try:
            results = self.model.val(data=data_config)
            logger.info("Validation completed")
            return results
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return None
