#!/usr/bin/env python3
"""
ResNet-50 Training Script for Quality Control Classification
Train custom ResNet model for defect classification
"""

import os
import yaml
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNetTrainer:
    """ResNet model trainer for quality control classification"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path)
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.writer = None
        self.best_accuracy = 0.0
        self.class_names = list(self.config['data']['classes'].values())
        
    def load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if self.config['hardware']['device'] == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config['hardware']['device'])
        
        logger.info(f"Using device: {device}")
        
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device
    
    def setup_data_loaders(self):
        """Setup data loaders with augmentation"""
        # Training transforms
        train_transforms = self._get_train_transforms()
        val_transforms = self._get_val_transforms()
        
        # Load datasets
        train_dataset = datasets.ImageFolder(
            root=self.config['data']['train_path'],
            transform=train_transforms
        )
        
        val_dataset = datasets.ImageFolder(
            root=self.config['data']['val_path'],
            transform=val_transforms
        )
        
        test_dataset = datasets.ImageFolder(
            root=self.config['data']['test_path'],
            transform=val_transforms
        )
        
        # Setup samplers for balanced training
        if self.config['data']['balance_dataset']:
            train_sampler = self._get_balanced_sampler(train_dataset)
        else:
            train_sampler = None
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['testing']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
    
    def _get_train_transforms(self) -> transforms.Compose:
        """Get training data transforms"""
        aug_config = self.config['augmentation']['train']
        
        transform_list = []
        
        # Resize and crop
        if 'random_resize_crop' in aug_config:
            rrc = aug_config['random_resize_crop']
            transform_list.append(transforms.RandomResizedCrop(
                rrc['size'],
                scale=rrc['scale'],
                ratio=rrc['ratio']
            ))
        else:
            transform_list.append(transforms.Resize(256))
            transform_list.append(transforms.CenterCrop(224))
        
        # Flip augmentations
        if aug_config.get('random_horizontal_flip', {}).get('probability', 0) > 0:
            transform_list.append(transforms.RandomHorizontalFlip(
                aug_config['random_horizontal_flip']['probability']
            ))
        
        if aug_config.get('random_vertical_flip', {}).get('probability', 0) > 0:
            transform_list.append(transforms.RandomVerticalFlip(
                aug_config['random_vertical_flip']['probability']
            ))
        
        # Rotation
        if aug_config.get('random_rotation', {}).get('degrees', 0) > 0:
            transform_list.append(transforms.RandomRotation(
                aug_config['random_rotation']['degrees']
            ))
        
        # Color jitter
        if 'color_jitter' in aug_config:
            cj = aug_config['color_jitter']
            transform_list.append(transforms.ColorJitter(
                brightness=cj.get('brightness', 0),
                contrast=cj.get('contrast', 0),
                saturation=cj.get('saturation', 0),
                hue=cj.get('hue', 0)
            ))
        
        # Convert to tensor and normalize
        transform_list.append(transforms.ToTensor())
        
        normalize = self.config['preprocessing']['normalize']
        transform_list.append(transforms.Normalize(
            mean=normalize['mean'],
            std=normalize['std']
        ))
        
        # Random erasing
        if aug_config.get('random_erasing', {}).get('enabled', False):
            re = aug_config['random_erasing']
            transform_list.append(transforms.RandomErasing(
                p=re['probability'],
                scale=re['scale'],
                ratio=re['ratio']
            ))
        
        return transforms.Compose(transform_list)
    
    def _get_val_transforms(self) -> transforms.Compose:
        """Get validation data transforms"""
        normalize = self.config['preprocessing']['normalize']
        
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
        ])
    
    def _get_balanced_sampler(self, dataset) -> WeightedRandomSampler:
        """Create balanced sampler for imbalanced datasets"""
        # Count samples per class
        class_counts = torch.zeros(len(self.class_names))
        for _, label in dataset:
            class_counts[label] += 1
        
        # Calculate weights
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for _, label in dataset]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def build_model(self):
        """Build ResNet model"""
        from torchvision import models
        
        # Load pretrained model
        if self.config['model']['architecture'] == 'resnet50':
            model = models.resnet50(pretrained=self.config['model']['pretrained'])
        elif self.config['model']['architecture'] == 'resnet101':
            model = models.resnet101(pretrained=self.config['model']['pretrained'])
        else:
            raise ValueError(f"Unsupported architecture: {self.config['model']['architecture']}")
        
        # Modify classifier
        num_features = model.fc.in_features
        classifier_config = self.config['architecture']['classifier']
        
        # Build custom classifier head
        layers = []
        input_features = num_features
        
        for hidden_size in classifier_config['hidden_layers']:
            layers.append(nn.Linear(input_features, hidden_size))
            
            if classifier_config['batch_norm']:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if classifier_config['activation'] == 'ReLU':
                layers.append(nn.ReLU(inplace=True))
            elif classifier_config['activation'] == 'GELU':
                layers.append(nn.GELU())
            
            if 'dropout' in classifier_config:
                dropout_idx = len(classifier_config['hidden_layers']) - len(layers) // 3
                if dropout_idx < len(classifier_config['dropout']):
                    layers.append(nn.Dropout(classifier_config['dropout'][dropout_idx]))
            
            input_features = hidden_size
        
        # Final classification layer
        layers.append(nn.Linear(input_features, self.config['model']['num_classes']))
        
        model.fc = nn.Sequential(*layers)
        
        # Freeze layers if specified
        if self.config['architecture']['freeze_layers'] > 0:
            self._freeze_layers(model, self.config['architecture']['freeze_layers'])
        
        self.model = model.to(self.device)
        logger.info(f"Model built: {self.config['model']['architecture']}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def _freeze_layers(self, model, num_layers: int):
        """Freeze specified number of layers"""
        layers = list(model.children())[:-1]  # Exclude final FC layer
        
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
        
        logger.info(f"Frozen {num_layers} layers")
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, and loss function"""
        # Setup optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if self.config['training']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                momentum=self.config['training']['momentum'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        
        # Setup scheduler
        scheduler_config = self.config['training']['lr_scheduler']
        if scheduler_config['type'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_config['type'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=scheduler_config['patience'],
                factor=scheduler_config['gamma']
            )
        
        # Setup loss function
        if 'class_weights' in self.config['loss']:
            class_weights = torch.tensor(self.config['loss']['class_weights']).to(self.device)
        else:
            class_weights = None
        
        if self.config['loss'].get('focal_loss', {}).get('enabled', False):
            # Implement focal loss if needed
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            label_smoothing = self.config['loss'].get('label_smoothing', 0.0)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
        
        # Setup TensorBoard
        if self.config['logging']['tensorboard']['enabled']:
            log_dir = os.path.join(
                self.config['logging']['log_dir'],
                self.config['logging']['experiment_name'],
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logs: {log_dir}")
    
    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.config['training']['mixed_precision']:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Use gradient scaler for mixed precision
                if not hasattr(self, 'scaler'):
                    self.scaler = torch.cuda.amp.GradScaler()
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress
            if batch_idx % self.config['logging']['print_frequency'] == 0:
                logger.info(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                          f'Loss: {loss.item():.4f} '
                          f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> tuple:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return val_loss, val_acc, precision, recall, f1
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        best_acc = 0.0
        patience_counter = 0
        early_stopping_patience = self.config['training']['early_stopping']['patience']
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config['validation']['val_frequency'] == 0:
                val_loss, val_acc, precision, recall, f1 = self.validate(epoch)
                
                # Log metrics
                if self.writer:
                    self.writer.add_scalar('Loss/Train', train_loss, epoch)
                    self.writer.add_scalar('Loss/Val', val_loss, epoch)
                    self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                    self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
                    self.writer.add_scalar('Precision/Val', precision, epoch)
                    self.writer.add_scalar('Recall/Val', recall, epoch)
                    self.writer.add_scalar('F1/Val', f1, epoch)
                    self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
                
                logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                          f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_checkpoint(epoch, val_acc, is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Update learning rate scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Save periodic checkpoint
            if epoch % self.config['checkpointing']['save_frequency'] == 0:
                self.save_checkpoint(epoch, val_acc)
        
        logger.info(f"Training completed. Best validation accuracy: {best_acc:.2f}%")
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, epoch: int, accuracy: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config['checkpointing']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'accuracy': accuracy,
            'config': self.config
        }
        
        if is_best:
            filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        else:
            filepath = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def test(self, model_path: str = None):
        """Test the model"""
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        test_acc = 100. * correct / total
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        self.plot_confusion_matrix(cm)
        
        return test_acc
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['checkpointing']['checkpoint_dir'], 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved: {plot_path}")
        plt.show()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ResNet for QC classification')
    parser.add_argument('--config', type=str, default='config/resnet_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true', help='Run testing only')
    parser.add_argument('--test_model', type=str, help='Path to model for testing')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ResNetTrainer(args.config)
        
        # Setup components
        trainer.setup_data_loaders()
        trainer.build_model()
        trainer.setup_training_components()
        
        if args.test:
            # Test mode
            test_acc = trainer.test(args.test_model)
            print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
        else:
            # Training mode
            if args.resume:
                # Resume training from checkpoint
                checkpoint = torch.load(args.resume, map_location=trainer.device)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if trainer.scheduler and checkpoint['scheduler_state_dict']:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"Resumed training from {args.resume}")
            
            # Start training
            trainer.train()
            
            # Test with best model
            best_model_path = os.path.join(
                trainer.config['checkpointing']['checkpoint_dir'],
                'best_model.pth'
            )
            if os.path.exists(best_model_path):
                test_acc = trainer.test(best_model_path)
                print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
