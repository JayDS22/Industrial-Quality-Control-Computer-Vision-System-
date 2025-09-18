#!/usr/bin/env python3
"""
Training Utilities for Quality Control Models
Common functions and classes for model training
"""

import os
import random
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import yaml
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def save_config(config: dict, path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {path}")

def load_config(path: str) -> dict:
    """Load configuration from file"""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'max'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop early"""
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement"""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_best(self, metric: str) -> Tuple[float, int]:
        """Get best value and epoch for a metric"""
        if metric not in self.metrics or not self.metrics[metric]:
            return None, None
        
        values = self.metrics[metric]
        if 'loss' in metric:
            best_value = min(values)
            best_epoch = values.index(best_value)
        else:
            best_value = max(values)
            best_epoch = values.index(best_value)
        
        return best_value, best_epoch
    
    def plot_metrics(self, save_path: str = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.metrics['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.metrics['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision, Recall, F1
        axes[1, 0].plot(self.metrics['val_precision'], label='Precision')
        axes[1, 0].plot(self.metrics['val_recall'], label='Recall')
        axes[1, 0].plot(self.metrics['val_f1'], label='F1 Score')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.metrics['learning_rate'])
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()
    
    def save_to_json(self, path: str):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {path}")

class ModelCheckpoint:
    """Model checkpointing utility"""
    
    def __init__(self, 
                 filepath: str,
                 monitor: str = 'val_acc',
                 mode: str = 'max',
                 save_best_only: bool = True,
                 save_frequency: int = 1):
        """
        Initialize model checkpoint
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'max' or 'min'
            save_best_only: Only save best model
            save_frequency: Save every N epochs
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.best_score = None
        self.epoch_count = 0
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def __call__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 epoch: int,
                 metrics: dict) -> bool:
        """Save checkpoint if criteria met"""
        self.epoch_count += 1
        score = metrics.get(self.monitor)
        
        if score is None:
            logger.warning(f"Metric {self.monitor} not found in metrics")
            return False
        
        is_best = self._is_best_score(score)
        should_save = (not self.save_best_only or is_best) and \
                     (self.epoch_count % self.save_frequency == 0)
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': metrics,
                'best_score': self.best_score
            }
            
            if is_best:
                filepath = self.filepath.replace('.pth', '_best.pth')
                torch.save(checkpoint, filepath)
                logger.info(f"Best model saved: {filepath} (score: {score:.4f})")
            else:
                filepath = self.filepath.replace('.pth', f'_epoch_{epoch}.pth')
                torch.save(checkpoint, filepath)
                logger.info(f"Checkpoint saved: {filepath}")
            
            return True
        
        return False
    
    def _is_best_score(self, score: float) -> bool:
        """Check if score is the best so far"""
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max' and score > self.best_score:
            self.best_score = score
            return True
        elif self.mode == 'min' and score < self.best_score:
            self.best_score = score
            return True
        
        return False

def calculate_class_weights(dataset_path: str, num_classes: int) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets"""
    from torchvision import datasets
    
    dataset = datasets.ImageFolder(dataset_path)
    class_counts = torch.zeros(num_classes)
    
    for _, label in dataset:
        class_counts[label] += 1
    
    # Calculate inverse frequency weights
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    
    logger.info(f"Class counts: {class_counts.tolist()}")
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    return class_weights

def plot_class_distribution(dataset_path: str, class_names: List[str], save_path: str = None):
    """Plot class distribution in dataset"""
    from torchvision import datasets
    
    dataset = datasets.ImageFolder(dataset_path)
    class_counts = [0] * len(class_names)
    
    for _, label in dataset:
        class_counts[label] += 1
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_counts)
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(class_counts),
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()

def plot_roc_curves(y_true: np.ndarray, 
                   y_scores: np.ndarray, 
                   class_names: List[str],
                   save_path: str = None):
    """Plot ROC curves for multi-class classification"""
    n_classes = len(class_names)
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    else:
        plt.show()

def create_training_report(model_name: str,
                          config: dict,
                          metrics: dict,
                          test_results: dict,
                          save_path: str):
    """Create comprehensive training report"""
    report = {
        'model_info': {
            'name': model_name,
            'timestamp': datetime.now().isoformat(),
            'config': config
        },
        'training_results': {
            'final_metrics': {
                'train_accuracy': metrics.get('train_acc', [])[-1] if metrics.get('train_acc') else None,
                'val_accuracy': metrics.get('val_acc', [])[-1] if metrics.get('val_acc') else None,
                'val_precision': metrics.get('val_precision', [])[-1] if metrics.get('val_precision') else None,
                'val_recall': metrics.get('val_recall', [])[-1] if metrics.get('val_recall') else None,
                'val_f1': metrics.get('val_f1', [])[-1] if metrics.get('val_f1') else None
            },
            'best_metrics': {
                'best_val_acc': max(metrics.get('val_acc', [0])),
                'best_val_precision': max(metrics.get('val_precision', [0])),
                'best_val_recall': max(metrics.get('val_recall', [0])),
                'best_val_f1': max(metrics.get('val_f1', [0]))
            }
        },
        'test_results': test_results,
        'performance_summary': {
            'meets_accuracy_target': test_results.get('accuracy', 0) >= config.get('targets', {}).get('overall_accuracy', 0.9) * 100,
            'meets_precision_target': test_results.get('precision', 0) >= config.get('targets', {}).get('precision', 0.9),
            'meets_recall_target': test_results.get('recall', 0) >= config.get('targets', {}).get('recall', 0.9)
        }
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to {save_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {test_results.get('accuracy', 0):.2f}%")
    print(f"Test Precision: {test_results.get('precision', 0):.4f}")
    print(f"Test Recall: {test_results.get('recall', 0):.4f}")
    print(f"Test F1 Score: {test_results.get('f1', 0):.4f}")
    print("="*50)

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def profile_model(model: torch.nn.Module, 
                 input_shape: Tuple[int, ...],
                 device: torch.device,
                 num_runs: int = 100) -> Dict[str, float]:
    """Profile model inference speed"""
    model.eval()
    
    # Warm up
    dummy_input = torch.randn(1, *input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    times = []
    
    for _ in range(num_runs):
        if device.type == 'cuda':
            start_time.record()
            with torch.no_grad():
                _ = model(dummy_input)
            end_time.record()
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))
        else:
            import time
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'throughput_fps': 1000.0 / np.mean(times)
    }
