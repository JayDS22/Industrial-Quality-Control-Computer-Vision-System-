#!/usr/bin/env python3
"""
YOLOv8 Training Script for Quality Control
Train custom YOLO model for defect detection
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOTrainer:
    """YOLO model trainer for quality control"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path)
        self.model = None
        self.training_results = None
        
    def load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Return default training configuration"""
        return {
            'model': {
                'architecture': 'yolov8n.pt',
                'input_size': 640
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.01,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'patience': 50,
                'save_period': 10
            },
            'data': {
                'train_path': 'datasets/qc_dataset/train',
                'val_path': 'datasets/qc_dataset/val',
                'test_path': 'datasets/qc_dataset/test',
                'classes': ['crack', 'scratch', 'dent', 'discoloration', 'contamination']
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0
            }
        }
    
    def prepare_dataset(self):
        """Prepare dataset configuration file"""
        logger.info("Preparing dataset configuration...")
        
        # Create dataset.yaml for YOLO training
        dataset_config = {
            'path': os.path.abspath('datasets/qc_dataset'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.config['data']['classes']),
            'names': self.config['data']['classes']
        }
        
        # Save dataset configuration
        dataset_path = 'datasets/qc_dataset/dataset.yaml'
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        with open(dataset_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset configuration saved to {dataset_path}")
        return dataset_path
    
    def initialize_model(self):
        """Initialize YOLO model"""
        try:
            model_arch = self.config['model']['architecture']
            logger.info(f"Initializing YOLO model: {model_arch}")
            
            self.model = YOLO(model_arch)
            logger.info("YOLO model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def train(self, dataset_path: str):
        """Train the YOLO model"""
        if self.model is None:
            self.initialize_model()
        
        try:
            logger.info("Starting YOLO training...")
            
            # Training parameters
            train_config = self.config['training']
            aug_config = self.config['augmentation']
            
            # Start training
            self.training_results = self.model.train(
                data=dataset_path,
                epochs=train_config['epochs'],
                batch=train_config['batch_size'],
                imgsz=self.config['model']['input_size'],
                lr0=train_config['learning_rate'],
                weight_decay=train_config['weight_decay'],
                momentum=train_config['momentum'],
                patience=train_config['patience'],
                save_period=train_config['save_period'],
                device='0' if torch.cuda.is_available() else 'cpu',
                workers=8,
                project='runs/train',
                name='qc_yolo',
                exist_ok=True,
                pretrained=True,
                optimizer='SGD',
                verbose=True,
                seed=42,
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=False,
                close_mosaic=10,
                resume=False,
                amp=True,
                fraction=1.0,
                profile=False,
                # Data augmentation parameters
                hsv_h=aug_config['hsv_h'],
                hsv_s=aug_config['hsv_s'],
                hsv_v=aug_config['hsv_v'],
                degrees=aug_config['degrees'],
                translate=aug_config['translate'],
                scale=aug_config['scale'],
                shear=aug_config['shear'],
                perspective=aug_config['perspective'],
                flipud=aug_config['flipud'],
                fliplr=aug_config['fliplr'],
                mosaic=aug_config['mosaic'],
                mixup=aug_config['mixup'],
                copy_paste=0.0
            )
            
            logger.info("Training completed successfully")
            return self.training_results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def validate(self, dataset_path: str):
        """Validate the trained model"""
        if self.model is None:
            logger.error("No model to validate")
            return None
        
        try:
            logger.info("Starting model validation...")
            
            # Run validation
            validation_results = self.model.val(
                data=dataset_path,
                batch=self.config['training']['batch_size'],
                imgsz=self.config['model']['input_size'],
                device='0' if torch.cuda.is_available() else 'cpu',
                save_json=True,
                save_hybrid=False,
                conf=0.001,
                iou=0.6,
                max_det=300,
                half=True,
                dnn=False,
                plots=True
            )
            
            logger.info("Validation completed")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return None
    
    def test(self, dataset_path: str, model_path: str = None):
        """Test the model on test dataset"""
        try:
            # Load best model if path provided
            if model_path:
                self.model = YOLO(model_path)
                logger.info(f"Loaded model from {model_path}")
            
            logger.info("Starting model testing...")
            
            # Run testing
            test_results = self.model.val(
                data=dataset_path,
                split='test',
                batch=self.config['training']['batch_size'],
                imgsz=self.config['model']['input_size'],
                device='0' if torch.cuda.is_available() else 'cpu',
                save_json=True,
                save_hybrid=False,
                conf=0.25,
                iou=0.45,
                max_det=300,
                half=True,
                plots=True
            )
            
            logger.info("Testing completed")
            return test_results
            
        except Exception as e:
            logger.error(f"Error during testing: {str(e)}")
            return None
    
    def export_model(self, export_path: str, format: str = 'onnx'):
        """Export trained model"""
        if self.model is None:
            logger.error("No model to export")
            return False
        
        try:
            logger.info(f"Exporting model to {format} format...")
            
            # Export model
            exported_model = self.model.export(
                format=format,
                imgsz=self.config['model']['input_size'],
                keras=False,
                optimize=False,
                half=False,
                int8=False,
                dynamic=False,
                simplify=False,
                opset=None,
                workspace=4,
                nms=False
            )
            
            # Copy to specified path
            if export_path:
                import shutil
                shutil.copy2(exported_model, export_path)
                logger.info(f"Model exported to {export_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            return False
    
    def plot_training_results(self, save_path: str = None):
        """Plot training results"""
        if self.training_results is None:
            logger.warning("No training results to plot")
            return
        
        try:
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('YOLO Training Results', fontsize=16)
            
            # Plot training metrics (if available in results)
            # Note: Actual plotting would depend on the structure of training_results
            # This is a placeholder implementation
            
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            
            axes[0, 1].set_title('Validation mAP')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP@0.5')
            
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training plots saved to {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
    
    def save_model(self, save_path: str):
        """Save trained model"""
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train YOLO model for QC')
    parser.add_argument('--config', type=str, default='config/yolo_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--data', type=str, help='Path to dataset YAML file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--export', action='store_true', help='Export model after training')
    parser.add_argument('--test', action='store_true', help='Run testing after training')
    parser.add_argument('--plot', action='store_true', help='Plot training results')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = YOLOTrainer(args.config)
        
        # Prepare dataset if not provided
        if not args.data:
            dataset_path = trainer.prepare_dataset()
        else:
            dataset_path = args.data
        
        # Train model
        logger.info("Starting YOLO training pipeline...")
        training_results = trainer.train(dataset_path)
        
        # Validate model
        validation_results = trainer.validate(dataset_path)
        
        # Test model if requested
        if args.test:
            test_results = trainer.test(dataset_path)
        
        # Export model if requested
        if args.export:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"models/yolov8_qc_{timestamp}.onnx"
            trainer.export_model(export_path, 'onnx')
        
        # Plot results if requested
        if args.plot:
            plot_path = f"runs/train/qc_yolo/training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            trainer.plot_training_results(plot_path)
        
        # Save final model
        final_model_path = "models/yolov8_qc.pt"
        trainer.save_model(final_model_path)
        
        logger.info("Training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        if validation_results:
            print(f"Final mAP@0.5: {validation_results.box.map50:.4f}")
            print(f"Final mAP@0.5-0.95: {validation_results.box.map:.4f}")
            print(f"Precision: {validation_results.box.mp:.4f}")
            print(f"Recall: {validation_results.box.mr:.4f}")
        print(f"Model saved to: {final_model_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
