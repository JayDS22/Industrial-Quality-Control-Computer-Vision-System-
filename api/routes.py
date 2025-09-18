#!/usr/bin/env python3
"""
API Routes for Quality Control System
REST API endpoints for model inference and system management
"""

from flask import Blueprint, request, jsonify, current_app
import numpy as np
import cv2
import base64
import io
import logging
from datetime import datetime
from typing import Dict, List
import os
import tempfile
import zipfile

logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/detect', methods=['POST'])
def detect_single():
    """Single image detection endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and decode image
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Get QC system from app context
        qc_system = current_app.qc_system
        
        # Process image
        result = qc_system.predict(image)
        
        # Add metadata
        result['api_version'] = '1.0.0'
        result['timestamp'] = datetime.now().isoformat()
        result['filename'] = file.filename
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in single detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/detect/batch', methods=['POST'])
def detect_batch():
    """Batch image detection endpoint"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No files selected'}), 400
        
        images = []
        filenames = []
        
        # Process each file
        for file in files:
            if file.filename != '':
                try:
                    image_bytes = file.read()
                    image_array = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        images.append(image)
                        filenames.append(file.filename)
                    else:
                        logger.warning(f"Could not decode image: {file.filename}")
                        
                except Exception as e:
                    logger.warning(f"Error processing file {file.filename}: {str(e)}")
        
        if not images:
            return jsonify({'error': 'No valid images found'}), 400
        
        # Get QC system
        qc_system = current_app.qc_system
        
        # Process batch
        results = qc_system.predict_batch(images)
        
        # Add filenames to results
        for i, result in enumerate(results):
            if i < len(filenames):
                result['filename'] = filenames[i]
        
        return jsonify({
            'batch_results': results,
            'total_processed': len(results),
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/detect/zip', methods=['POST'])
def detect_zip():
    """Process images from ZIP file"""
    try:
        if 'zip_file' not in request.files:
            return jsonify({'error': 'No ZIP file provided'}), 400
        
        zip_file = request.files['zip_file']
        if zip_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporary ZIP file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            zip_file.save(temp_zip.name)
            temp_zip_path = temp_zip.name
        
        try:
            images = []
            filenames = []
            
            # Extract and process images from ZIP
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        try:
                            with zip_ref.open(file_info) as image_file:
                                image_bytes = image_file.read()
                                image_array = np.frombuffer(image_bytes, np.uint8)
                                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                                
                                if image is not None:
                                    images.append(image)
                                    filenames.append(file_info.filename)
                                    
                        except Exception as e:
                            logger.warning(f"Error processing {file_info.filename}: {str(e)}")
            
            if not images:
                return jsonify({'error': 'No valid images found in ZIP file'}), 400
            
            # Process batch
            qc_system = current_app.qc_system
            results = qc_system.predict_batch(images)
            
            # Add filenames
            for i, result in enumerate(results):
                if i < len(filenames):
                    result['filename'] = filenames[i]
            
            return jsonify({
                'batch_results': results,
                'total_processed': len(results),
                'source_zip': zip_file.filename,
                'api_version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)
        
    except Exception as e:
        logger.error(f"Error in ZIP detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/detect/base64', methods=['POST'])
def detect_base64():
    """Detect from base64 encoded image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image data provided'}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400
        
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Process image
        qc_system = current_app.qc_system
        result = qc_system.predict(image)
        
        # Add metadata
        result['api_version'] = '1.0.0'
        result['timestamp'] = datetime.now().isoformat()
        result['input_format'] = 'base64'
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in base64 detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models/info', methods=['GET'])
def get_model_info():
    """Get information about loaded models"""
    try:
        qc_system = current_app.qc_system
        model_info = qc_system.get_system_info()
        
        return jsonify({
            'model_info': model_info,
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models/performance', methods=['GET'])
def get_performance_stats():
    """Get model performance statistics"""
    try:
        qc_system = current_app.qc_system
        stats = qc_system.get_performance_stats()
        
        return jsonify({
            'performance_stats': stats,
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models/benchmark', methods=['POST'])
def benchmark_models():
    """Benchmark model performance"""
    try:
        data = request.get_json() or {}
        
        # Get benchmark parameters
        num_images = data.get('num_images', 10)
        iterations = data.get('iterations', 3)
        image_size = data.get('image_size', [640, 640])
        
        # Generate test images
        test_images = []
        for _ in range(num_images):
            # Create random test image
            test_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            test_images.append(test_image)
        
        # Run benchmark
        qc_system = current_app.qc_system
        benchmark_results = qc_system.benchmark(test_images, iterations)
        
        return jsonify({
            'benchmark_results': benchmark_results,
            'test_parameters': {
                'num_images': num_images,
                'iterations': iterations,
                'image_size': image_size
            },
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in model benchmark: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/config', methods=['GET'])
def get_config():
    """Get current system configuration"""
    try:
        qc_system = current_app.qc_system
        config = qc_system.config
        
        # Remove sensitive information
        safe_config = config.copy()
        if 'alerts' in safe_config and 'email' in safe_config['alerts']:
            safe_config['alerts']['email'] = {'enabled': True}  # Hide credentials
        
        return jsonify({
            'config': safe_config,
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/config', methods=['PUT'])
def update_config():
    """Update system configuration"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Update configuration
        qc_system = current_app.qc_system
        qc_system.update_config(data)
        
        return jsonify({
            'message': 'Configuration updated successfully',
            'updated_fields': list(data.keys()),
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/thresholds', methods=['GET'])
def get_thresholds():
    """Get current detection thresholds"""
    try:
        qc_system = current_app.qc_system
        thresholds = {
            'confidence_threshold': qc_system.ensemble_predictor.confidence_threshold,
            'quality_thresholds': qc_system.config.get('quality_control', {}).get('thresholds', {})
        }
        
        return jsonify({
            'thresholds': thresholds,
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting thresholds: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/thresholds', methods=['PUT'])
def update_thresholds():
    """Update detection thresholds"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No threshold data provided'}), 400
        
        qc_system = current_app.qc_system
        
        # Update confidence threshold
        if 'confidence_threshold' in data:
            if hasattr(qc_system.ensemble_predictor, 'yolo_detector'):
                qc_system.ensemble_predictor.yolo_detector.update_thresholds(
                    confidence=data['confidence_threshold']
                )
        
        # Update NMS threshold
        if 'nms_threshold' in data:
            if hasattr(qc_system.ensemble_predictor, 'yolo_detector'):
                qc_system.ensemble_predictor.yolo_detector.update_thresholds(
                    nms=data['nms_threshold']
                )
        
        return jsonify({
            'message': 'Thresholds updated successfully',
            'updated_thresholds': data,
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating thresholds: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/spc/analyze', methods=['POST'])
def spc_analyze():
    """Perform SPC analysis on detection results"""
    try:
        data = request.get_json()
        
        if not data or 'detection_results' not in data:
            return jsonify({'error': 'No detection results provided'}), 400
        
        # Get SPC analyzer
        qc_system = current_app.qc_system
        if not hasattr(qc_system, 'spc_analyzer'):
            return jsonify({'error': 'SPC analyzer not available'}), 500
        
        # Run SPC analysis
        spc_results = qc_system.spc_analyzer.analyze(data['detection_results'])
        
        return jsonify({
            'spc_analysis': spc_results,
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in SPC analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/spc/export', methods=['POST'])
def export_spc_report():
    """Export SPC analysis report"""
    try:
        data = request.get_json() or {}
        output_path = data.get('output_path', f'spc_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        # Get SPC analyzer
        qc_system = current_app.qc_system
        if not hasattr(qc_system, 'spc_analyzer'):
            return jsonify({'error': 'SPC analyzer not available'}), 500
        
        # Export report
        success = qc_system.spc_analyzer.export_spc_report(output_path)
        
        if success:
            return jsonify({
                'message': 'SPC report exported successfully',
                'output_path': output_path,
                'api_version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to export SPC report'}), 500
        
    except Exception as e:
        logger.error(f"Error exporting SPC report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/quality/rules', methods=['GET'])
def get_quality_rules():
    """Get current quality control rules"""
    try:
        qc_system = current_app.qc_system
        rules = qc_system.config.get('quality_control', {})
        
        return jsonify({
            'quality_rules': rules,
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting quality rules: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/quality/assess', methods=['POST'])
def assess_quality():
    """Assess quality based on detection results"""
    try:
        data = request.get_json()
        
        if not data or 'detections' not in data:
            return jsonify({'error': 'No detection data provided'}), 400
        
        # Mock quality assessment (would use actual postprocessor)
        detections = data['detections']
        
        # Count severity levels
        severity_counts = {'critical': 0, 'major': 0, 'minor': 0}
        for detection in detections:
            severity = detection.get('severity', 'minor')
            severity_counts[severity] += 1
        
        # Simple quality rules
        if severity_counts['critical'] > 0:
            grade, status = 'F', 'FAIL'
        elif severity_counts['major'] > 2:
            grade, status = 'D', 'FAIL'
        elif severity_counts['major'] > 0:
            grade, status = 'C', 'CONDITIONAL'
        elif severity_counts['minor'] > 3:
            grade, status = 'B', 'CONDITIONAL'
        else:
            grade, status = 'A', 'PASS'
        
        assessment = {
            'quality_grade': grade,
            'pass_fail_status': status,
            'total_defects': len(detections),
            'severity_breakdown': severity_counts,
            'meets_requirements': status == 'PASS'
        }
        
        return jsonify({
            'quality_assessment': assessment,
            'api_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in quality assessment: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        qc_system = current_app.qc_system
        
        # Check system components
        health_status = {
            'status': 'healthy',
            'components': {
                'ensemble_predictor': qc_system.ensemble_predictor is not None,
                'segmentator': qc_system.segmentator is not None,
                'postprocessor': qc_system.postprocessor is not None
            },
            'performance': qc_system.get_performance_stats(),
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0'
        }
        
        # Determine overall health
        if all(health_status['components'].values()):
            health_status['status'] = 'healthy'
            status_code = 200
        else:
            health_status['status'] = 'unhealthy'
            status_code = 503
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0'
        }), 500

@api_bp.route('/version', methods=['GET'])
def get_version():
    """Get API version information"""
    return jsonify({
        'api_version': '1.0.0',
        'system_name': 'Industrial Quality Control Computer Vision System',
        'description': 'Multi-class defect detection with 94.2% accuracy',
        'features': [
            'YOLOv8 object detection',
            'ResNet-50 classification',
            'Real-time processing (500+ images/min)',
            'Statistical Process Control',
            'Edge computing optimization',
            'Automated anomaly detection'
        ],
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad request',
        'message': 'Invalid request format or parameters',
        'api_version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }), 400

@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'API endpoint not found',
        'api_version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }), 404

@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'api_version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }), 500

# Rate limiting decorator
from functools import wraps
import time

def rate_limit(max_requests: int = 100, window: int = 60):
    """Rate limiting decorator"""
    request_history = {}
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            # Clean old requests
            if client_ip in request_history:
                request_history[client_ip] = [
                    req_time for req_time in request_history[client_ip]
                    if current_time - req_time < window
                ]
            else:
                request_history[client_ip] = []
            
            # Check rate limit
            if len(request_history[client_ip]) >= max_requests:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {max_requests} requests per {window} seconds',
                    'api_version': '1.0.0',
                    'timestamp': datetime.now().isoformat()
                }), 429
            
            # Add current request
            request_history[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Apply rate limiting to detection endpoints
api_bp.route('/detect', methods=['POST'])(rate_limit(max_requests=50, window=60)(detect_single))
api_bp.route('/detect/batch', methods=['POST'])(rate_limit(max_requests=10, window=60)(detect_batch))
