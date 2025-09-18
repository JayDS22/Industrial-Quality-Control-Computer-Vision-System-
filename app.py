#!/usr/bin/env python3
"""
Industrial Quality Control Computer Vision System
Main Flask Application

Author: Your Name
Date: 2024
"""

import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import yaml
import numpy as np
from datetime import datetime
import threading
import queue
import time

# Import custom modules
from models.yolo_model import YOLODetector
from models.resnet_model import ResNetClassifier
from models.ensemble import EnsemblePredictor
from inference.detector import QualityControlDetector
from analytics.spc_analysis import SPCAnalyzer
from analytics.anomaly_detection import AnomalyDetector
from api.routes import api_bp
from dashboard.dashboard import dashboard_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QualityControlSystem:
    """Main Quality Control System Class"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config = self.load_config(config_path)
        self.detector = None
        self.spc_analyzer = None
        self.anomaly_detector = None
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_processing = False
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            'model': {
                'yolo_weights': 'models/yolov8_qc.pt',
                'resnet_weights': 'models/resnet50_qc.pt',
                'confidence_threshold': 0.7,
                'nms_threshold': 0.5
            },
            'processing': {
                'batch_size': 8,
                'max_workers': 4,
                'enable_gpu': True,
                'tensorrt_optimization': True
            },
            'quality_control': {
                'defect_classes': ['crack', 'scratch', 'dent', 'discoloration', 'contamination'],
                'severity_levels': ['minor', 'major', 'critical']
            }
        }
    
    def initialize_models(self):
        """Initialize all AI models"""
        try:
            logger.info("Initializing AI models...")
            
            # Initialize main detector
            self.detector = QualityControlDetector(
                yolo_weights=self.config['model']['yolo_weights'],
                resnet_weights=self.config['model']['resnet_weights'],
                config=self.config
            )
            
            # Initialize analytics modules
            self.spc_analyzer = SPCAnalyzer()
            self.anomaly_detector = AnomalyDetector()
            
            logger.info("All models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            return False
    
    def start_processing_worker(self):
        """Start background processing worker"""
        def worker():
            while True:
                try:
                    if not self.processing_queue.empty():
                        task = self.processing_queue.get()
                        result = self.process_image(task['image'], task['metadata'])
                        self.results_queue.put(result)
                    time.sleep(0.01)  # Small delay to prevent CPU overload
                except Exception as e:
                    logger.error(f"Processing worker error: {str(e)}")
        
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        logger.info("Processing worker started")
    
    def process_image(self, image, metadata=None):
        """Process a single image through the QC pipeline"""
        try:
            start_time = time.time()
            
            # Run detection and classification
            results = self.detector.predict(image)
            
            # Perform SPC analysis
            spc_results = self.spc_analyzer.analyze(results)
            
            # Check for anomalies
            anomaly_score = self.anomaly_detector.detect(results)
            
            processing_time = time.time() - start_time
            
            return {
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'spc_analysis': spc_results,
                'anomaly_score': anomaly_score,
                'processing_time_ms': processing_time * 1000,
                'metadata': metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize QC System
qc_system = QualityControlSystem()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': qc_system.detector is not None
    })

@app.route('/api/detect', methods=['POST'])
def detect_single():
    """Single image detection endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_bytes = file.read()
        image = np.frombuffer(image_bytes, np.uint8)
        
        # Process image
        result = qc_system.process_image(image, {'filename': file.filename})
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in detect_single: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_detect', methods=['POST'])
def detect_batch():
    """Batch image detection endpoint"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            if file.filename != '':
                image_bytes = file.read()
                image = np.frombuffer(image_bytes, np.uint8)
                result = qc_system.process_image(image, {'filename': file.filename})
                results.append(result)
        
        return jsonify({
            'batch_results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in detect_batch: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_statistics():
    """Get system statistics"""
    return jsonify({
        'system_status': 'operational',
        'models_loaded': qc_system.detector is not None,
        'queue_size': qc_system.processing_queue.qsize(),
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    emit('status', {'message': 'Connected to QC System'})
    logger.info('Client connected via WebSocket')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info('Client disconnected from WebSocket')

@socketio.on('start_realtime')
def handle_realtime_start():
    """Start real-time processing"""
    qc_system.is_processing = True
    emit('status', {'message': 'Real-time processing started'})
    logger.info('Real-time processing started')

@socketio.on('stop_realtime')
def handle_realtime_stop():
    """Stop real-time processing"""
    qc_system.is_processing = False
    emit('status', {'message': 'Real-time processing stopped'})
    logger.info('Real-time processing stopped')

def initialize_system():
    """Initialize the complete system"""
    logger.info("Starting Industrial QC System...")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    
    # Initialize models
    if not qc_system.initialize_models():
        logger.warning("Models not initialized - system running in demo mode")
    
    # Start background workers
    qc_system.start_processing_worker()
    
    logger.info("System initialization complete")

if __name__ == '__main__':
    initialize_system()
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    
    # Run the application
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        allow_unsafe_werkzeug=True
    )
