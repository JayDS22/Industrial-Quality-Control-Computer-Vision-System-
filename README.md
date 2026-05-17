# Industrial Quality Control Computer Vision System

## Overview

A comprehensive computer vision system for automated quality control in manufacturing environments, combining YOLOv8 object detection, ResNet-50 classification, and real-time processing capabilities for industrial-grade defect detection.

## Key Features

- **Multi-class defect detection** with 94.2% accuracy
- **Real-time processing** of 500+ images/minute
- **Edge computing optimization** with TensorRT
- **Statistical Process Control (SPC)** dashboard
- **Automated anomaly detection**
- **Production-ready Flask API**

## Performance Metrics

- **Overall Accuracy**: 94.2%
- **mAP@0.5**: 88.7%
- **Precision**: 91.3%
- **Processing Speed**: 500+ images/minute
- **Inference Time**: <150ms
- **Model Size Reduction**: 60% (with TensorRT)

## Architecture

```mermaid
flowchart LR
    classDef in fill:#1d2a3a,stroke:#58a6ff,stroke-width:2px,color:#e6edf3
    classDef ml fill:#1f2a23,stroke:#3fb950,stroke-width:2px,color:#e6edf3
    classDef edge fill:#1a1a2e,stroke:#c084fc,stroke-width:2px,color:#e6edf3
    classDef out fill:#2a2520,stroke:#c9a227,stroke-width:2px,color:#e6edf3

    I[Input<br/>camera feed · files · batch]:::in
    P[Preprocessing<br/>augment → normalize → resize → QC]:::ml
    M[Detection + Classification<br/>YOLOv8 + ResNet-50]:::ml
    E[Edge Optimization<br/>TensorRT · INT8 quant · CUDA]:::edge
    A[Analytics<br/>segmentation · SPC · anomaly]:::ml
    API[Flask REST + WebSocket]:::out
    O[Dashboard + Alerts<br/>real-time · SPC charts · email/SMS]:::out

    I --> P --> M --> E --> A --> API --> O

    click M href "models" "ML models"
    click P href "data_processing" "Preprocessing"
    click E href "inference" "Inference engine"
    click A href "analytics" "Analytics"
    click API href "api" "REST API"
    click O href "dashboard" "Dashboard"
```

### System Components

1. **Input Processing**
   - Multi-source image ingestion (cameras, files, batches)
   - Real-time streaming capabilities
   - Quality validation and preprocessing

2. **AI/ML Pipeline**
   - **YOLOv8**: Object detection for defect localization
   - **ResNet-50**: Classification for defect categorization
   - Ensemble inference for improved accuracy

3. **Edge Optimization**
   - TensorRT model optimization
   - CUDA acceleration
   - Memory-efficient processing

4. **Analytics Engine**
   - Statistical Process Control (SPC)
   - Real-time anomaly detection
   - Trend analysis and reporting

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- TensorRT (for edge deployment)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/industrial-qc-cv-system.git
   cd industrial-qc-cv-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models**
   ```bash
   python scripts/download_models.py
   ```

4. **Configure environment**
   ```bash
   cp config/config.example.yaml config/config.yaml
   # Edit config.yaml with your settings
   ```

## Quick Start

### Training Models

1. **Prepare your dataset**
   ```bash
   python scripts/prepare_dataset.py --data_path /path/to/your/data
   ```

2. **Train YOLOv8 detection model**
   ```bash
   python train/train_yolo.py --config config/yolo_config.yaml
   ```

3. **Train ResNet-50 classification model**
   ```bash
   python train/train_resnet.py --config config/resnet_config.yaml
   ```

### Running the System

1. **Start the Flask API**
   ```bash
   python app.py
   ```

2. **Access the dashboard**
   - Open browser: `http://localhost:5000`
   - Upload images or connect camera feed

3. **API Usage**
   ```bash
   # Single image processing
   curl -X POST -F "image=@test_image.jpg" http://localhost:5000/api/detect
   
   # Batch processing
   curl -X POST -F "images=@batch.zip" http://localhost:5000/api/batch_detect
   ```

## Project Structure

```
industrial-qc-cv-system/
├── README.md
├── requirements.txt
├── app.py                      # Flask application
├── config/
│   ├── config.yaml            # Main configuration
│   ├── yolo_config.yaml       # YOLO training config
│   └── resnet_config.yaml     # ResNet training config
├── models/
│   ├── __init__.py
│   ├── yolo_model.py          # YOLOv8 implementation
│   ├── resnet_model.py        # ResNet-50 implementation
│   ├── ensemble.py            # Model ensemble
│   └── tensorrt_optimizer.py  # TensorRT optimization
├── train/
│   ├── train_yolo.py          # YOLO training script
│   ├── train_resnet.py        # ResNet training script
│   └── utils.py               # Training utilities
├── inference/
│   ├── detector.py            # Main detection engine
│   ├── segmentation.py        # Image segmentation
│   └── postprocess.py         # Post-processing
├── data_processing/
│   ├── augmentation.py        # Data augmentation
│   ├── preprocessing.py       # Image preprocessing
│   └── dataset_loader.py      # Dataset handling
├── analytics/
│   ├── spc_analysis.py        # Statistical Process Control
│   ├── anomaly_detection.py   # Anomaly detection
│   └── quality_metrics.py     # Quality calculations
├── api/
│   ├── routes.py              # API endpoints
│   ├── websocket.py           # Real-time communication
│   └── middleware.py          # API middleware
├── dashboard/
│   ├── static/                # CSS, JS files
│   ├── templates/             # HTML templates
│   └── dashboard.py           # Dashboard logic
├── scripts/
│   ├── download_models.py     # Model download script
│   ├── prepare_dataset.py     # Dataset preparation
│   └── benchmark.py           # Performance benchmarking
├── tests/
│   ├── test_models.py         # Model tests
│   ├── test_api.py            # API tests
│   └── test_integration.py    # Integration tests
└── deployment/
    ├── docker/                # Docker configurations
    ├── kubernetes/            # K8s deployment files
    └── edge/                  # Edge deployment scripts
```

## Configuration

Edit `config/config.yaml`:

```yaml
model:
  yolo_weights: "models/yolov8_qc.pt"
  resnet_weights: "models/resnet50_qc.pt"
  confidence_threshold: 0.7
  nms_threshold: 0.5

processing:
  batch_size: 8
  max_workers: 4
  enable_gpu: true
  tensorrt_optimization: true

quality_control:
  defect_classes:
    - "crack"
    - "scratch" 
    - "dent"
    - "discoloration"
    - "contamination"
  severity_levels:
    - "minor"
    - "major"
    - "critical"

alerts:
  email_notifications: true
  sms_notifications: false
  webhook_url: "https://your-webhook.com/alerts"
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_models.py -v
python -m pytest tests/test_api.py -v

# Performance benchmarking
python scripts/benchmark.py
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t industrial-qc-system .

# Run container
docker run -p 5000:5000 --gpus all industrial-qc-system
```

### Edge Deployment

```bash
# Optimize models for edge
python deployment/edge/optimize_for_edge.py

# Deploy to edge device
python deployment/edge/deploy.py --device jetson_nano
```

## Performance Monitoring

The system includes comprehensive monitoring:

- Real-time processing metrics
- Model accuracy tracking
- System resource utilization
- Quality control statistics
- Alert management

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- ResNet by Microsoft Research
- TensorRT by NVIDIA
- OpenCV community


---

**Built with for Industrial Quality Control**
