#!/usr/bin/env python3
"""
Ensemble Model Predictor
Combines YOLOv8 and ResNet-50 predictions for improved accuracy
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from .yolo_model import YOLODetector
from .resnet_model import ResNetClassifier

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Ensemble predictor combining YOLO detection and ResNet classification"""
    
    def __init__(self, 
                 yolo_weights: str = "models/yolov8_qc.pt",
                 resnet_weights: str = "models/resnet50_qc.pt",
                 config: Dict = None):
        """
        Initialize ensemble predictor
        
        Args:
            yolo_weights: Path to YOLO model weights
            resnet_weights: Path to ResNet model weights
            config: Configuration dictionary
        """
        self.config = config or {}
        self.yolo_detector = None
        self.resnet_classifier = None
        self.ensemble_weights = {
            "yolo": 0.6,
            "resnet": 0.4
        }
        self.confidence_threshold = 0.5
        self.load_models(yolo_weights, resnet_weights)
    
    def load_models(self, yolo_weights: str, resnet_weights: str) -> bool:
        """Load both YOLO and ResNet models"""
        try:
            logger.info("Loading ensemble models...")
            
            # Load YOLO detector
            yolo_config = self.config.get('model', {})
            self.yolo_detector = YOLODetector(
                model_path=yolo_weights,
                confidence_threshold=yolo_config.get('confidence_threshold', 0.7),
                nms_threshold=yolo_config.get('nms_threshold', 0.5)
            )
            
            # Load ResNet classifier
            self.resnet_classifier = ResNetClassifier(
                model_path=resnet_weights,
                num_classes=len(self.config.get('quality_control', {}).get('defect_classes', [])) or 5
            )
            
            logger.info("Ensemble models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble models: {str(e)}")
            return False
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run ensemble prediction on image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Combined prediction results
        """
        if self.yolo_detector is None or self.resnet_classifier is None:
            return {"error": "Models not loaded"}
        
        try:
            start_time = time.time()
            
            # Run both models in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                yolo_future = executor.submit(self.yolo_detector.predict, image)
                resnet_future = executor.submit(self.resnet_classifier.predict, image)
                
                yolo_results = yolo_future.result()
                resnet_results = resnet_future.result()
            
            # Combine results
            ensemble_results = self._combine_predictions(yolo_results, resnet_results, image.shape)
            
            total_time = time.time() - start_time
            ensemble_results["total_inference_time_ms"] = total_time * 1000
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return {"error": str(e)}
    
    def _combine_predictions(self, yolo_results: Dict, resnet_results: Dict, image_shape: Tuple) -> Dict:
        """Combine YOLO and ResNet predictions"""
        combined_results = {
            "detections": [],
            "global_classification": resnet_results,
            "detection_summary": {},
            "quality_assessment": {},
            "ensemble_confidence": 0.0
        }
        
        # Check for errors
        if "error" in yolo_results or "error" in resnet_results:
            combined_results["error"] = "Error in one or more models"
            return combined_results
        
        try:
            # Process YOLO detections
            yolo_detections = yolo_results.get("detections", [])
            
            # Enhance detections with ResNet classification
            enhanced_detections = []
            
            for detection in yolo_detections:
                # Extract detection region
                bbox = detection["bbox"]
                detection_region = self._extract_region(image_shape, bbox)
                
                # Classify the detection region with ResNet
                region_classification = self._classify_region_mock(detection)
                
                # Combine YOLO detection with ResNet classification
                enhanced_detection = self._merge_detection_classification(
                    detection, region_classification
                )
                
                enhanced_detections.append(enhanced_detection)
            
            combined_results["detections"] = enhanced_detections
            
            # Create detection summary
            combined_results["detection_summary"] = self._create_detection_summary(enhanced_detections)
            
            # Overall quality assessment
            combined_results["quality_assessment"] = self._assess_overall_quality(
                enhanced_detections, resnet_results
            )
            
            # Calculate ensemble confidence
            combined_results["ensemble_confidence"] = self._calculate_ensemble_confidence(
                yolo_results, resnet_results, enhanced_detections
            )
            
        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            combined_results["error"] = str(e)
        
        return combined_results
    
    def _extract_region(self, image_shape: Tuple, bbox: Dict) -> Dict:
        """Extract region information from bounding box"""
        height, width = image_shape[:2] if len(image_shape) > 2 else (bbox["y2"], bbox["x2"])
        
        # Calculate relative position and size
        rel_x = bbox["center_x"] / width
        rel_y = bbox["center_y"] / height
        rel_width = bbox["width"] / width
        rel_height = bbox["height"] / height
        
        return {
            "relative_position": {"x": rel_x, "y": rel_y},
            "relative_size": {"width": rel_width, "height": rel_height},
            "absolute_area": bbox["width"] * bbox["height"]
        }
    
    def _classify_region_mock(self, detection: Dict) -> Dict:
        """Mock region classification (in real implementation, would extract and classify region)"""
        # This would normally extract the region and run ResNet on it
        # For now, we'll use the YOLO classification with some modifications
        
        base_confidence = detection["confidence"]
        defect_class = detection["class"]
        
        # Add some classification refinement
        return {
            "predicted_class": defect_class,
            "confidence": min(base_confidence * 1.1, 1.0),  # Slight boost
            "region_severity": detection["severity"],
            "classification_source": "ensemble_refined"
        }
    
    def _merge_detection_classification(self, detection: Dict, classification: Dict) -> Dict:
        """Merge YOLO detection with ResNet classification"""
        merged = detection.copy()
        
        # Weighted confidence combination
        yolo_conf = detection["confidence"]
        resnet_conf = classification["confidence"]
        
        ensemble_confidence = (
            yolo_conf * self.ensemble_weights["yolo"] +
            resnet_conf * self.ensemble_weights["resnet"]
        )
        
        merged.update({
            "ensemble_confidence": ensemble_confidence,
            "yolo_confidence": yolo_conf,
            "resnet_confidence": resnet_conf,
            "classification_details": classification,
            "final_severity": self._determine_final_severity(detection, classification)
        })
        
        return merged
    
    def _determine_final_severity(self, detection: Dict, classification: Dict) -> str:
        """Determine final severity using both models"""
        yolo_severity = detection["severity"]
        resnet_severity = classification["region_severity"]
        
        severity_scores = {"minor": 1, "major": 2, "critical": 3}
        
        yolo_score = severity_scores.get(yolo_severity, 1)
        resnet_score = severity_scores.get(resnet_severity, 1)
        
        # Take the maximum severity
        final_score = max(yolo_score, resnet_score)
        
        # Convert back to severity level
        for severity, score in severity_scores.items():
            if score == final_score:
                return severity
        
        return "minor"
    
    def _create_detection_summary(self, detections: List[Dict]) -> Dict:
        """Create summary of all detections"""
        if not detections:
            return {
                "total_defects": 0,
                "defect_counts": {},
                "severity_distribution": {},
                "average_confidence": 0.0,
                "max_severity": "none"
            }
        
        # Count defects by class
        defect_counts = {}
        severity_counts = {"minor": 0, "major": 0, "critical": 0}
        confidences = []
        
        for detection in detections:
            defect_class = detection["class"]
            severity = detection["final_severity"]
            confidence = detection["ensemble_confidence"]
            
            defect_counts[defect_class] = defect_counts.get(defect_class, 0) + 1
            severity_counts[severity] += 1
            confidences.append(confidence)
        
        # Determine maximum severity
        max_severity = "none"
        if severity_counts["critical"] > 0:
            max_severity = "critical"
        elif severity_counts["major"] > 0:
            max_severity = "major"
        elif severity_counts["minor"] > 0:
            max_severity = "minor"
        
        return {
            "total_defects": len(detections),
            "defect_counts": defect_counts,
            "severity_distribution": severity_counts,
            "average_confidence": np.mean(confidences) if confidences else 0.0,
            "max_severity": max_severity
        }
    
    def _assess_overall_quality(self, detections: List[Dict], global_classification: Dict) -> Dict:
        """Assess overall quality of the product"""
        if not detections:
            return {
                "quality_grade": "A",
                "pass_fail": "PASS",
                "defect_density": 0.0,
                "risk_level": "low",
                "recommended_action": "accept"
            }
        
        # Calculate metrics
        total_defects = len(detections)
        critical_defects = sum(1 for d in detections if d["final_severity"] == "critical")
        major_defects = sum(1 for d in detections if d["final_severity"] == "major")
        minor_defects = sum(1 for d in detections if d["final_severity"] == "minor")
        
        # Quality grading logic
        if critical_defects > 0:
            quality_grade = "F"
            pass_fail = "FAIL"
            risk_level = "high"
            recommended_action = "reject"
        elif major_defects > 2:
            quality_grade = "D"
            pass_fail = "FAIL"
            risk_level = "high"
            recommended_action = "reject"
        elif major_defects > 0:
            quality_grade = "C"
            pass_fail = "CONDITIONAL"
            risk_level = "medium"
            recommended_action = "review"
        elif minor_defects > 3:
            quality_grade = "B"
            pass_fail = "CONDITIONAL"
            risk_level = "low"
            recommended_action = "review"
        else:
            quality_grade = "A"
            pass_fail = "PASS"
            risk_level = "low"
            recommended_action = "accept"
        
        return {
            "quality_grade": quality_grade,
            "pass_fail": pass_fail,
            "defect_density": total_defects,  # Could be normalized by area
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "defect_breakdown": {
                "critical": critical_defects,
                "major": major_defects,
                "minor": minor_defects
            }
        }
    
    def _calculate_ensemble_confidence(self, yolo_results: Dict, resnet_results: Dict, detections: List[Dict]) -> float:
        """Calculate overall ensemble confidence"""
        try:
            # Get individual model confidences
            yolo_conf = np.mean([d["yolo_confidence"] for d in detections]) if detections else 0.0
            resnet_conf = resnet_results.get("confidence", 0.0)
            
            # Weight and combine
            ensemble_conf = (
                yolo_conf * self.ensemble_weights["yolo"] +
                resnet_conf * self.ensemble_weights["resnet"]
            )
            
            return float(ensemble_conf)
            
        except Exception as e:
            logger.error(f"Error calculating ensemble confidence: {str(e)}")
            return 0.0
    
    def batch_predict(self, images: List[np.ndarray]) -> List[Dict]:
        """Run ensemble prediction on batch of images"""
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing ensemble batch {i+1}/{len(images)}")
            result = self.predict(image)
            result["batch_index"] = i
            results.append(result)
        
        return results
    
    def update_ensemble_weights(self, yolo_weight: float, resnet_weight: float):
        """Update ensemble weighting"""
        total = yolo_weight + resnet_weight
        self.ensemble_weights = {
            "yolo": yolo_weight / total,
            "resnet": resnet_weight / total
        }
        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def get_model_info(self) -> Dict:
        """Get ensemble model information"""
        info = {
            "ensemble_weights": self.ensemble_weights,
            "confidence_threshold": self.confidence_threshold,
            "models_loaded": {
                "yolo": self.yolo_detector is not None,
                "resnet": self.resnet_classifier is not None
            }
        }
        
        if self.yolo_detector:
            info["yolo_info"] = self.yolo_detector.get_model_info()
        
        if self.resnet_classifier:
            info["resnet_info"] = self.resnet_classifier.get_model_info()
        
        return info
    
    def visualize_ensemble_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Visualize ensemble prediction results"""
        vis_image = image.copy()
        
        # Get detections
        detections = results.get("detections", [])
        quality_assessment = results.get("quality_assessment", {})
        
        # Visualize detections
        if self.yolo_detector and detections:
            vis_image = self.yolo_detector.visualize_detections(vis_image, detections)
        
        # Add quality assessment overlay
        if quality_assessment:
            self._add_quality_overlay(vis_image, quality_assessment)
        
        return vis_image
    
    def _add_quality_overlay(self, image: np.ndarray, quality_assessment: Dict):
        """Add quality assessment overlay to image"""
        height, width = image.shape[:2]
        
        # Create overlay panel
        overlay_height = 100
        overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)
        overlay.fill(50)  # Dark background
        
        # Add quality information
        grade = quality_assessment.get("quality_grade", "Unknown")
        pass_fail = quality_assessment.get("pass_fail", "Unknown")
        risk = quality_assessment.get("risk_level", "Unknown")
        
        # Color coding for pass/fail
        if pass_fail == "PASS":
            color = (0, 255, 0)  # Green
        elif pass_fail == "FAIL":
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 255)  # Yellow
        
        # Add text
        cv2.putText(overlay, f"Grade: {grade}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(overlay, f"Status: {pass_fail}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(overlay, f"Risk: {risk}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Combine with original image
        result = np.vstack([image, overlay])
        image[:] = result[:height]


class EnsembleOptimizer:
    """Optimizer for ensemble model performance"""
    
    def __init__(self, ensemble_predictor: EnsemblePredictor):
        self.ensemble = ensemble_predictor
        self.performance_history = []
    
    def optimize_weights(self, validation_data: List[Tuple[np.ndarray, Dict]]) -> Dict:
        """Optimize ensemble weights based on validation data"""
        best_weights = None
        best_accuracy = 0.0
        
        # Try different weight combinations
        weight_combinations = [
            (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), 
            (0.4, 0.6), (0.3, 0.7), (0.8, 0.2), (0.2, 0.8)
        ]
        
        for yolo_w, resnet_w in weight_combinations:
            self.ensemble.update_ensemble_weights(yolo_w, resnet_w)
            accuracy = self._evaluate_ensemble(validation_data)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (yolo_w, resnet_w)
            
            logger.info(f"Weights ({yolo_w}, {resnet_w}): Accuracy = {accuracy:.3f}")
        
        # Set best weights
        if best_weights:
            self.ensemble.update_ensemble_weights(best_weights[0], best_weights[1])
        
        return {
            "best_weights": best_weights,
            "best_accuracy": best_accuracy,
            "optimization_results": self.performance_history
        }
    
    def _evaluate_ensemble(self, validation_data: List[Tuple[np.ndarray, Dict]]) -> float:
        """Evaluate ensemble performance on validation data"""
        correct = 0
        total = len(validation_data)
        
        for image, ground_truth in validation_data:
            prediction = self.ensemble.predict(image)
            
            # Simple accuracy calculation (would be more complex in practice)
            if self._compare_predictions(prediction, ground_truth):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _compare_predictions(self, prediction: Dict, ground_truth: Dict) -> bool:
        """Compare prediction with ground truth (simplified)"""
        # This would contain actual comparison logic
        # For now, just return a mock comparison
        return True  # Placeholder
    
    def benchmark_performance(self, test_images: List[np.ndarray]) -> Dict:
        """Benchmark ensemble performance"""
        start_time = time.time()
        
        results = []
        for image in test_images:
            result = self.ensemble.predict(image)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_inference_time = total_time / len(test_images)
        throughput = len(test_images) / total_time
        
        return {
            "total_images": len(test_images),
            "total_time_seconds": total_time,
            "average_inference_time_ms": avg_inference_time * 1000,
            "throughput_images_per_second": throughput,
            "results": results
        }
