
#!/usr/bin/env python3
"""
Comprehensive Test Suite for Industrial QC Models
Tests for YOLO, ResNet, and Ensemble components
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
import yaml
from unittest.mock import Mock, patch
import logging

# Import modules to test
from models.yolo_model import YOLODetector, YOLOTrainer
from models.resnet_model import ResNetClassifier, ResNetTrainer
from models.ensemble import EnsemblePredictor, EnsembleOptimizer
from inference.detector import QualityControlDetector

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestYOLODetector:
    """Test suite for YOLO detector"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'model': {
                'confidence_threshold': 0.7,
                'nms_threshold': 0.5
            }
        }
    
    @pytest.fixture
    def sample_image(self):
        """Generate sample test image"""
        # Create a simple test image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        return image
    
    @pytest.fixture
    def encoded_image(self):
        """Generate encoded test image"""
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', image)
        return encoded
    
    def test_detector_initialization(self, mock_config):
        """Test YOLO detector initialization"""
        with patch('models.yolo_model.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.names = {0: 'crack', 1: 'scratch'}
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(
                model_path="test_model.pt",
                confidence_threshold=0.7,
                device="cpu"
            )
            
            assert detector.confidence_threshold == 0.7
            assert detector.device == "cpu"
            assert detector.model is not None
    
    def test_image_preprocessing(self, sample_image):
        """Test image preprocessing functionality"""
        with patch('models.yolo_model.YOLO'):
            detector = YOLODetector(device="cpu")
            
            # Test RGB image
            processed = detector.preprocess_image(sample_image)
            assert processed.shape == sample_image.shape
            
            # Test encoded image
            _, encoded = cv2.imencode('.jpg', sample_image)
            processed_encoded = detector.preprocess_image(encoded)
            assert len(processed_encoded.shape) == 3
    
    def test_prediction_with_mock_results(self, sample_image):
        """Test prediction with mocked YOLO results"""
        with patch('models.yolo_model.YOLO') as mock_yolo:
            # Setup mock model and results
            mock_model = Mock()
            mock_result = Mock()
            mock_boxes = Mock()
            
            # Mock detection data
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
            mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
            mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
            
            mock_result.boxes = mock_boxes
            mock_model.return_value = [mock_result]
            mock_model.names = {0: 'crack'}
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(device="cpu")
            result = detector.predict(sample_image)
            
            assert "detections" in result
            assert "inference_time_ms" in result
            assert len(result["detections"]) > 0
            assert result["detections"][0]["class"] == "crack"
    
    def test_batch_prediction(self, sample_image):
        """Test batch prediction functionality"""
        with patch('models.yolo_model.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.return_value = [Mock()]
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(device="cpu")
            images = [sample_image, sample_image]
            results = detector.batch_predict(images)
            
            assert len(results) == 2
            assert all("batch_index" in result for result in results)
    
    def test_severity_classification(self):
        """Test defect severity classification"""
        with patch('models.yolo_model.YOLO'):
            detector = YOLODetector(device="cpu")
            
            # Test different confidence and area combinations
            assert detector._classify_severity(0.95, 100000) == "critical"
            assert detector._classify_severity(0.85, 50000) == "major"
            assert detector._classify_severity(0.7, 1000) == "minor"
    
    def test_visualization(self, sample_image):
        """Test detection visualization"""
        with patch('models.yolo_model.YOLO'):
            detector = YOLODetector(device="cpu")
            
            # Mock detections
            detections = [{
                "class": "crack",
                "confidence": 0.8,
                "severity": "major",
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            }]
            
            vis_image = detector.visualize_detections(sample_image, detections)
            assert vis_image.shape == sample_image.shape
            # Check that visualization was applied (image should be modified)
            assert not np.array_equal(vis_image, sample_image)


class TestResNetClassifier:
    """Test suite for ResNet classifier"""
    
    @pytest.fixture
    def sample_image(self):
        """Generate sample test image"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_classifier_initialization(self):
        """Test ResNet classifier initialization"""
        with patch('models.resnet_model.models.resnet50'):
            classifier = ResNetClassifier(
                model_path="test_model.pt",
                num_classes=5,
                device="cpu"
            )
            
            assert classifier.num_classes == 5
            assert len(classifier.class_names) == 5
    
    def test_image_preprocessing(self, sample_image):
        """Test image preprocessing"""
        with patch('models.resnet_model.models.resnet50'):
            classifier = ResNetClassifier(device="cpu")
            
            processed = classifier.preprocess_image(sample_image)
            assert processed.shape[0] == 1  # Batch dimension
            assert processed.shape[1] == 3  # Channels
            assert processed.shape[2] == 224  # Height
            assert processed.shape[3] == 224  # Width
    
    def test_prediction_with_mock_results(self, sample_image):
        """Test prediction with mocked PyTorch results"""
        with patch('models.resnet_model.models.resnet50') as mock_resnet:
            with patch('torch.load'):
                mock_model = Mock()
                mock_resnet.return_value = mock_model
                
                # Mock prediction results
                with patch.object(mock_model, '__call__') as mock_call:
                    import torch
                    mock_output = torch.tensor([[2.0, 1.0, 0.5, 0.2, 0.1]])
                    mock_call.return_value = mock_output
                    
                    classifier = ResNetClassifier(device="cpu")
                    classifier.model = mock_model
                    
                    result = classifier.predict(sample_image)
                    
                    assert "predicted_class" in result
                    assert "confidence" in result
                    assert "class_probabilities" in result
                    assert "severity" in result
    
    def test_batch_prediction(self, sample_image):
        """Test batch prediction"""
        with patch('models.resnet_model.models.resnet50'):
            classifier = ResNetClassifier(device="cpu")
            classifier.model = Mock()
            
            # Mock batch results
            import torch
            with patch.object(classifier.model, '__call__') as mock_call:
                mock_output = torch.tensor([[2.0, 1.0, 0.5, 0.2, 0.1],
                                           [1.0, 2.0, 0.3, 0.1, 0.2]])
                mock_call.return_value = mock_output
                
                images = [sample_image, sample_image]
                results = classifier.predict_batch(images)
                
                assert len(results) == 2
                assert all("batch_index" in result for result in results)
    
    def test_feature_extraction(self, sample_image):
        """Test feature extraction functionality"""
        with patch('models.resnet_model.models.resnet50'):
            classifier = ResNetClassifier(device="cpu")
            
            # Mock feature extraction
            mock_features = Mock()
            mock_features.cpu.return_value.numpy.return_value = np.random.rand(1, 2048)
            
            with patch.object(classifier.model, 'features') as mock_feat:
                with patch.object(classifier.model, 'avgpool') as mock_pool:
                    with patch('torch.flatten') as mock_flatten:
                        mock_flatten.return_value = mock_features
                        
                        features = classifier.extract_features(sample_image)
                        assert features is not None
                        assert len(features.shape) == 1


class TestEnsemblePredictor:
    """Test suite for ensemble predictor"""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'quality_control': {
                'defect_classes': ['crack', 'scratch', 'dent', 'discoloration', 'contamination']
            }
        }
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    def test_ensemble_initialization(self, mock_config):
        """Test ensemble predictor initialization"""
        with patch('models.ensemble.YOLODetector') as mock_yolo:
            with patch('models.ensemble.ResNetClassifier') as mock_resnet:
                ensemble = EnsemblePredictor(
                    yolo_weights="test_yolo.pt",
                    resnet_weights="test_resnet.pt",
                    config=mock_config
                )
                
                assert ensemble.config == mock_config
                assert ensemble.yolo_detector is not None
                assert ensemble.resnet_classifier is not None
    
    def test_prediction_combination(self, mock_config, sample_image):
        """Test prediction combination logic"""
        with patch('models.ensemble.YOLODetector') as mock_yolo_class:
            with patch('models.ensemble.ResNetClassifier') as mock_resnet_class:
                # Setup mocks
                mock_yolo = Mock()
                mock_resnet = Mock()
                mock_yolo_class.return_value = mock_yolo
                mock_resnet_class.return_value = mock_resnet
                
                # Mock results
                yolo_result = {
                    "detections": [{
                        "class": "crack",
                        "confidence": 0.8,
                        "severity": "major",
                        "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, 
                               "width": 40, "height": 40, "center_x": 30, "center_y": 30}
                    }],
                    "inference_time_ms": 100
                }
                
                resnet_result = {
                    "predicted_class": "crack",
                    "confidence": 0.9,
                    "severity": "major"
                }
                
                mock_yolo.predict.return_value = yolo_result
                mock_resnet.predict.return_value = resnet_result
                
                ensemble = EnsemblePredictor(config=mock_config)
                result = ensemble.predict(sample_image)
                
                assert "detections" in result
                assert "global_classification" in result
                assert "quality_assessment" in result
                assert "ensemble_confidence" in result
    
    def test_ensemble_weights_update(self, mock_config):
        """Test ensemble weights updating"""
        with patch('models.ensemble.YOLODetector'):
            with patch('models.ensemble.ResNetClassifier'):
                ensemble = EnsemblePredictor(config=mock_config)
                
                ensemble.update_ensemble_weights(0.7, 0.3)
                assert ensemble.ensemble_weights["yolo"] == 0.7
                assert ensemble.ensemble_weights["resnet"] == 0.3
    
    def test_quality_assessment(self, mock_config):
        """Test quality assessment logic"""
        with patch('models.ensemble.YOLODetector'):
            with patch('models.ensemble.ResNetClassifier'):
                ensemble = EnsemblePredictor(config=mock_config)
                
                # Test different defect scenarios
                test_cases = [
                    # No defects - should pass
                    ([], "A", "PASS"),
                    # Minor defects only - should pass
                    ([{"final_severity": "minor"}] * 2, "B", "CONDITIONAL"),
                    # Major defects - should fail
                    ([{"final_severity": "major"}] * 3, "D", "FAIL"),
                    # Critical defects - should fail
                    ([{"final_severity": "critical"}], "F", "FAIL")
                ]
                
                for detections, expected_grade, expected_pass_fail in test_cases:
                    assessment = ensemble._assess_overall_quality(detections, {})
                    assert assessment["quality_grade"] == expected_grade
                    assert assessment["pass_fail"] == expected_pass_fail


class TestQualityControlDetector:
    """Test suite for main QC detector"""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'processing': {
                'batch_size': 8,
                'max_workers': 4
            },
            'preprocessing': {
                'resize': [640, 640],
                'denoise': False,
                'enhance_contrast': False
            }
        }
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, mock_config):
        """Test QC detector initialization"""
        with patch('inference.detector.EnsemblePredictor') as mock_ensemble:
            with patch('inference.detector.ImageSegmentator') as mock_seg:
                with patch('inference.detector.PostProcessor') as mock_post:
                    detector = QualityControlDetector(
                        yolo_weights="test_yolo.pt",
                        resnet_weights="test_resnet.pt",
                        config=mock_config
                    )
                    
                    assert detector.config == mock_config
                    assert detector.ensemble_predictor is not None
                    assert detector.segmentator is not None
                    assert detector.postprocessor is not None
    
    def test_image_validation(self, mock_config, sample_image):
        """Test image validation logic"""
        with patch('inference.detector.EnsemblePredictor'):
            with patch('inference.detector.ImageSegmentator'):
                with patch('inference.detector.PostProcessor'):
                    detector = QualityControlDetector(
                        yolo_weights="test_yolo.pt",
                        resnet_weights="test_resnet.pt",
                        config=mock_config
                    )
                    
                    # Test valid image
                    assert detector._validate_image(sample_image) == True
                    
                    # Test invalid inputs
                    assert detector._validate_image(None) == False
                    assert detector._validate_image("not_an_image") == False
    
    def test_batch_prediction(self, mock_config, sample_image):
        """Test batch prediction functionality"""
        with patch('inference.detector.EnsemblePredictor') as mock_ensemble_class:
            with patch('inference.detector.ImageSegmentator'):
                with patch('inference.detector.PostProcessor') as mock_post_class:
                    # Setup mocks
                    mock_ensemble = Mock()
                    mock_post = Mock()
                    mock_ensemble_class.return_value = mock_ensemble
                    mock_post_class.return_value = mock_post
                    
                    # Mock prediction result
                    mock_result = {
                        "detections": [],
                        "ensemble_confidence": 0.8
                    }
                    mock_ensemble.predict.return_value = mock_result
                    mock_post.process_results.return_value = mock_result
                    
                    detector = QualityControlDetector(
                        yolo_weights="test_yolo.pt",
                        resnet_weights="test_resnet.pt",
                        config=mock_config
                    )
                    
                    images = [sample_image, sample_image, sample_image]
                    results = detector.predict_batch(images, max_workers=2)
                    
                    assert len(results) == 3
                    assert all("batch_index" in result for result in results)
                    assert all("batch_statistics" in result for result in results)
    
    def test_performance_stats(self, mock_config):
        """Test performance statistics tracking"""
        with patch('inference.detector.EnsemblePredictor'):
            with patch('inference.detector.ImageSegmentator'):
                with patch('inference.detector.PostProcessor'):
                    detector = QualityControlDetector(
                        yolo_weights="test_yolo.pt",
                        resnet_weights="test_resnet.pt",
                        config=mock_config
                    )
                    
                    # Test initial stats
                    stats = detector.get_performance_stats()
                    assert stats["total_predictions"] == 0
                    
                    # Simulate predictions
                    detector._update_performance_stats(0.1)
                    detector._update_performance_stats(0.2)
                    
                    updated_stats = detector.get_performance_stats()
                    assert updated_stats["total_predictions"] == 2
                    assert updated_stats["average_time"] == 0.15
    
    def test_benchmark(self, mock_config, sample_image):
        """Test benchmarking functionality"""
        with patch('inference.detector.EnsemblePredictor') as mock_ensemble_class:
            with patch('inference.detector.ImageSegmentator'):
                with patch('inference.detector.PostProcessor') as mock_post_class:
                    # Setup mocks
                    mock_ensemble = Mock()
                    mock_post = Mock()
                    mock_ensemble_class.return_value = mock_ensemble
                    mock_post_class.return_value = mock_post
                    
                    mock_result = {"detections": [], "ensemble_confidence": 0.8}
                    mock_ensemble.predict.return_value = mock_result
                    mock_post.process_results.return_value = mock_result
                    
                    detector = QualityControlDetector(
                        yolo_weights="test_yolo.pt",
                        resnet_weights="test_resnet.pt",
                        config=mock_config
                    )
                    
                    test_images = [sample_image] * 5
                    benchmark_results = detector.benchmark(test_images, iterations=2)
                    
                    assert "total_images" in benchmark_results
                    assert "timing_statistics" in benchmark_results
                    assert "throughput" in benchmark_results
                    assert benchmark_results["total_images"] == 10  # 5 images * 2 iterations


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary configuration file"""
        config = {
            'model': {
                'yolo_weights': 'test_yolo.pt',
                'resnet_weights': 'test_resnet.pt',
                'confidence_threshold': 0.7,
                'nms_threshold': 0.5
            },
            'processing': {
                'batch_size': 4,
                'max_workers': 2,
                'enable_gpu': False
            },
            'quality_control': {
                'defect_classes': ['crack', 'scratch', 'dent'],
                'severity_levels': ['minor', 'major', 'critical']
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_end_to_end_prediction(self, temp_config_file):
        """Test complete end-to-end prediction pipeline"""
        with patch('models.yolo_model.YOLO') as mock_yolo:
            with patch('models.resnet_model.models.resnet50') as mock_resnet:
                with patch('inference.segmentation.ImageSegmentator'):
                    with patch('inference.postprocess.PostProcessor') as mock_post_class:
                        # Setup mocks
                        mock_yolo_model = Mock()
                        mock_resnet_model = Mock()
                        mock_post = Mock()
                        
                        mock_yolo.return_value = mock_yolo_model
                        mock_resnet.return_value = mock_resnet_model
                        mock_post_class.return_value = mock_post
                        
                        # Mock YOLO results
                        mock_result = Mock()
                        mock_boxes = Mock()
                        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
                        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
                        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
                        mock_result.boxes = mock_boxes
                        mock_yolo_model.return_value = [mock_result]
                        mock_yolo_model.names = {0: 'crack'}
                        
                        # Mock ResNet results
                        import torch
                        mock_resnet_output = torch.tensor([[2.0, 1.0, 0.5]])
                        mock_resnet_model.return_value = mock_resnet_output
                        
                        # Mock post-processor
                        final_result = {
                            "detections": [{
                                "class": "crack",
                                "confidence": 0.8,
                                "severity": "major"
                            }],
                            "quality_assessment": {
                                "quality_grade": "C",
                                "pass_fail": "CONDITIONAL"
                            }
                        }
                        mock_post.process_results.return_value = final_result
                        
                        # Load config and create detector
                        with open(temp_config_file, 'r') as f:
                            config = yaml.safe_load(f)
                        
                        detector = QualityControlDetector(
                            yolo_weights=config['model']['yolo_weights'],
                            resnet_weights=config['model']['resnet_weights'],
                            config=config
                        )
                        
                        # Test prediction
                        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                        result = detector.predict(test_image)
                        
                        assert "detections" in result
                        assert "quality_assessment" in result
                        assert result["quality_assessment"]["quality_grade"] == "C"


class TestPerformance:
    """Performance and stress tests"""
    
    def test_memory_usage(self):
        """Test memory usage under load"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('models.yolo_model.YOLO'):
            with patch('models.resnet_model.models.resnet50'):
                # Create multiple detectors
                detectors = []
                for i in range(10):
                    detector = YOLODetector(device="cpu")
                    detectors.append(detector)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory increase should be reasonable (less than 1GB for 10 detectors)
                assert memory_increase < 1000, f"Memory usage too high: {memory_increase}MB"
                
                # Cleanup
                del detectors
                gc.collect()
    
    def test_concurrent_predictions(self):
        """Test concurrent prediction handling"""
        import threading
        import time
        
        results = []
        errors = []
        
        def predict_worker(worker_id):
            try:
                with patch('models.yolo_model.YOLO'):
                    detector = YOLODetector(device="cpu")
                    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    
                    # Mock the model call
                    detector.model = Mock()
                    mock_result = Mock()
                    mock_result.boxes = None
                    detector.model.return_value = [mock_result]
                    
                    result = detector.predict(test_image)
                    results.append((worker_id, result))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=predict_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        assert len(errors) == 0, f"Errors in concurrent execution: {errors}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory"""
    test_dir = tempfile.mkdtemp(prefix="qc_test_")
    yield test_dir
    # Cleanup would happen here in a real implementation


@pytest.fixture
def mock_model_files(test_data_dir):
    """Create mock model files for testing"""
    yolo_path = os.path.join(test_data_dir, "yolo_test.pt")
    resnet_path = os.path.join(test_data_dir, "resnet_test.pt")
    
    # Create empty files to simulate model files
    with open(yolo_path, 'w') as f:
        f.write("mock yolo model")
    with open(resnet_path, 'w') as f:
        f.write("mock resnet model")
    
    return yolo_path, resnet_path


def test_system_health():
    """Test overall system health and dependencies"""
    try:
        import torch
        import cv2
        import numpy as np
        import yaml
        import flask
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        # Check OpenCV
        cv2_version = cv2.__version__
        logger.info(f"OpenCV version: {cv2_version}")
        
        # Basic functionality test
        test_array = np.random.rand(10, 10, 3)
        assert test_array.shape == (10, 10, 3)
        
        logger.info("System health check passed")
        
    except ImportError as e:
        pytest.fail(f"Missing required dependency: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
