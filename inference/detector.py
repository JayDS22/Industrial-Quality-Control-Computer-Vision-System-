#!/usr/bin/env python3
"""
Main Quality Control Detector
Integrates all models and provides unified interface
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.ensemble import EnsemblePredictor
from inference.segmentation import ImageSegmentator
from inference.postprocess import PostProcessor

logger = logging.getLogger(__name__)

class QualityControlDetector:
    """Main quality control detection system"""
    
    def __init__(self, 
                 yolo_weights: str,
                 resnet_weights: str,
                 config: Dict):
        """
        Initialize QC detector
        
        Args:
            yolo_weights: Path to YOLO model
            resnet_weights: Path to ResNet model
            config: System configuration
        """
        self.config = config
        self.ensemble_predictor = None
        self.segmentator = None
        self.postprocessor = None
        self.performance_stats = {
            "total_predictions": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
        
        self.initialize_components(yolo_weights, resnet_weights)
    
    def initialize_components(self, yolo_weights: str, resnet_weights: str):
        """Initialize all detection components"""
        try:
            logger.info("Initializing QC detection components...")
            
            # Initialize ensemble predictor
            self.ensemble_predictor = EnsemblePredictor(
                yolo_weights=yolo_weights,
                resnet_weights=resnet_weights,
                config=self.config
            )
            
            # Initialize segmentation
            self.segmentator = ImageSegmentator(self.config)
            
            # Initialize post-processor
            self.postprocessor = PostProcessor(self.config)
            
            logger.info("QC detection components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing QC detector: {str(e)}")
            raise
    
    def predict(self, image: np.ndarray, include_segmentation: bool = True) -> Dict:
        """
        Main prediction method
        
        Args:
            image: Input image
            include_segmentation: Whether to include segmentation results
            
        Returns:
            Complete prediction results
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not self._validate_image(image):
                return {"error": "Invalid image input"}
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run ensemble prediction
            ensemble_results = self.ensemble_predictor.predict(processed_image)
            
            if "error" in ensemble_results:
                return ensemble_results
            
            # Run segmentation if requested
            segmentation_results = {}
            if include_segmentation and ensemble_results.get("detections"):
                segmentation_results = self.segmentator.segment_defects(
                    processed_image, ensemble_results["detections"]
                )
            
            # Post-process results
            final_results = self.postprocessor.process_results(
                ensemble_results, segmentation_results, processed_image.shape
            )
            
            # Update performance stats
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time)
            
            # Add metadata
            final_results.update({
                "total_inference_time_ms": inference_time * 1000,
                "processing_pipeline": "ensemble + segmentation + postprocess",
                "image_metadata": self._get_image_metadata(image)
            })
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in QC prediction: {str(e)}")
            return {"error": str(e)}
    
    def predict_batch(self, images: List[np.ndarray], max_workers: int = 4) -> List[Dict]:
        """
        Batch prediction with parallel processing
        
        Args:
            images: List of input images
            max_workers: Maximum number of worker threads
            
        Returns:
            List of prediction results
        """
        start_time = time.time()
        results = []
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all prediction tasks
                future_to_index = {
                    executor.submit(self.predict, img): i 
                    for i, img in enumerate(images)
                }
                
                # Collect results in order
                results = [None] * len(images)
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        result["batch_index"] = index
                        results[index] = result
                    except Exception as e:
                        logger.error(f"Error processing image {index}: {str(e)}")
                        results[index] = {"error": str(e), "batch_index": index}
            
            # Add batch statistics
            total_time = time.time() - start_time
            batch_stats = {
                "batch_size": len(images),
                "total_batch_time_ms": total_time * 1000,
                "average_time_per_image_ms": (total_time * 1000) / len(images),
                "throughput_images_per_second": len(images) / total_time
            }
            
            # Add batch stats to each result
            for result in results:
                if result:
                    result["batch_statistics"] = batch_stats
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return [{"error": str(e)} for _ in images]
    
    def predict_stream(self, image_generator, callback=None):
        """
        Stream prediction for real-time processing
        
        Args:
            image_generator: Generator yielding images
            callback: Optional callback function for results
        """
        try:
            for i, image in enumerate(image_generator):
                result = self.predict(image)
                result["stream_index"] = i
                result["timestamp"] = time.time()
                
                if callback:
                    callback(result)
                else:
                    yield result
                    
        except Exception as e:
            logger.error(f"Error in stream prediction: {str(e)}")
            if callback:
                callback({"error": str(e)})
            else:
                yield {"error": str(e)}
    
    def _validate_image(self, image: np.ndarray) -> bool:
        """Validate input image"""
        if image is None:
            return False
        
        if isinstance(image, np.ndarray):
            if len(image.shape) == 1:
                # Encoded image - try to decode
                try:
                    decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    return decoded is not None
                except:
                    return False
            elif len(image.shape) in [2, 3]:
                return True
        
        return False
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference"""
        if len(image.shape) == 1:
            # Decode if necessary
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply any configured preprocessing
        preprocessing_config = self.config.get('preprocessing', {})
        
        # Resize if configured
        if 'resize' in preprocessing_config:
            target_size = preprocessing_config['resize']
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Noise reduction
        if preprocessing_config.get('denoise', False):
            image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Contrast enhancement
        if preprocessing_config.get('enhance_contrast', False):
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image
    
    def _get_image_metadata(self, image: np.ndarray) -> Dict:
        """Extract image metadata"""
        if len(image.shape) == 1:
            # Encoded image
            decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
            shape = decoded.shape if decoded is not None else (0, 0, 0)
        else:
            shape = image.shape
        
        return {
            "original_shape": shape,
            "channels": shape[2] if len(shape) > 2 else 1,
            "dtype": str(image.dtype),
            "size_bytes": image.nbytes if hasattr(image, 'nbytes') else 0
        }
    
    def _update_performance_stats(self, inference_time: float):
        """Update performance statistics"""
        self.performance_stats["total_predictions"] += 1
        self.performance_stats["total_time"] += inference_time
        self.performance_stats["average_time"] = (
            self.performance_stats["total_time"] / 
            self.performance_stats["total_predictions"]
        )
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = self.performance_stats.copy()
        
        if stats["total_predictions"] > 0:
            stats.update({
                "average_time_ms": stats["average_time"] * 1000,
                "throughput_images_per_second": 1.0 / stats["average_time"] if stats["average_time"] > 0 else 0,
                "total_time_minutes": stats["total_time"] / 60
            })
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            "total_predictions": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
        logger.info("Performance statistics reset")
    
    def get_system_info(self) -> Dict:
        """Get complete system information"""
        info = {
            "detector_status": "operational",
            "components_loaded": {
                "ensemble_predictor": self.ensemble_predictor is not None,
                "segmentator": self.segmentator is not None,
                "postprocessor": self.postprocessor is not None
            },
            "performance_stats": self.get_performance_stats(),
            "configuration": self.config
        }
        
        if self.ensemble_predictor:
            info["ensemble_info"] = self.ensemble_predictor.get_model_info()
        
        return info
    
    def update_config(self, new_config: Dict):
        """Update system configuration"""
        self.config.update(new_config)
        
        # Update components if necessary
        if self.postprocessor:
            self.postprocessor.update_config(new_config)
        
        logger.info("Configuration updated")
    
    def benchmark(self, test_images: List[np.ndarray], iterations: int = 1) -> Dict:
        """Benchmark detector performance"""
        logger.info(f"Starting benchmark with {len(test_images)} images, {iterations} iterations")
        
        all_times = []
        all_results = []
        
        for iteration in range(iterations):
            start_time = time.time()
            
            for image in test_images:
                img_start = time.time()
                result = self.predict(image)
                img_time = time.time() - img_start
                
                all_times.append(img_time)
                all_results.append(result)
            
            iteration_time = time.time() - start_time
            logger.info(f"Iteration {iteration + 1}/{iterations}: {iteration_time:.2f}s")
        
        # Calculate statistics
        times_ms = [t * 1000 for t in all_times]
        
        benchmark_results = {
            "total_images": len(test_images) * iterations,
            "iterations": iterations,
            "timing_statistics": {
                "mean_ms": np.mean(times_ms),
                "median_ms": np.median(times_ms),
                "min_ms": np.min(times_ms),
                "max_ms": np.max(times_ms),
                "std_ms": np.std(times_ms),
                "p95_ms": np.percentile(times_ms, 95),
                "p99_ms": np.percentile(times_ms, 99)
            },
            "throughput": {
                "images_per_second": len(test_images) * iterations / sum(all_times),
                "images_per_minute": (len(test_images) * iterations / sum(all_times)) * 60
            },
            "accuracy_metrics": self._calculate_accuracy_metrics(all_results)
        }
        
        return benchmark_results
    
    def _calculate_accuracy_metrics(self, results: List[Dict]) -> Dict:
        """Calculate accuracy metrics from results"""
        total_results = len(results)
        successful_results = sum(1 for r in results if "error" not in r)
        
        total_detections = sum(
            len(r.get("detections", [])) 
            for r in results if "error" not in r
        )
        
        avg_confidence = np.mean([
            r.get("ensemble_confidence", 0.0) 
            for r in results if "error" not in r
        ]) if successful_results > 0 else 0.0
        
        return {
            "success_rate": successful_results / total_results if total_results > 0 else 0.0,
            "average_detections_per_image": total_detections / successful_results if successful_results > 0 else 0.0,
            "average_confidence": float(avg_confidence)
        }
