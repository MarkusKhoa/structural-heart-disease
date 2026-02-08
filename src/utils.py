"""
Utility functions for training, evaluation, and calibration.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional
import warnings


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics.
    
    Args:
        y_true: Ground truth labels (n_samples, n_labels)
        y_pred: Predicted probabilities (n_samples, n_labels)
        label_names: Optional list of label names
        
    Returns:
        Dictionary of metrics
    """
    n_labels = y_true.shape[1]
    
    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]
    
    metrics = {}
    
    # Compute per-label metrics
    for i, name in enumerate(label_names):
        # Skip if all labels are the same (no positive or no negative samples)
        if len(np.unique(y_true[:, i])) < 2:
            continue
            
        try:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auprc = average_precision_score(y_true[:, i], y_pred[:, i])
            
            metrics[f'{name}_auroc'] = auroc
            metrics[f'{name}_auprc'] = auprc
        except Exception as e:
            warnings.warn(f"Could not compute metrics for {name}: {e}")
    
    # Compute macro averages
    auroc_scores = [v for k, v in metrics.items() if 'auroc' in k]
    auprc_scores = [v for k, v in metrics.items() if 'auprc' in k]
    
    if auroc_scores:
        metrics['macro_auroc'] = np.mean(auroc_scores)
    if auprc_scores:
        metrics['macro_auprc'] = np.mean(auprc_scores)
    
    return metrics


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE).
    
    Args:
        y_true: Ground truth labels (n_samples, n_labels)
        y_pred: Predicted probabilities (n_samples, n_labels)
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    n_labels = y_true.shape[1]
    
    ece_scores = []
    mce_scores = []
    
    for i in range(n_labels):
        if len(np.unique(y_true[:, i])) < 2:
            continue
            
        try:
            # Compute calibration curve
            prob_true, prob_pred = calibration_curve(
                y_true[:, i], y_pred[:, i], 
                n_bins=n_bins, strategy='uniform'
            )
            
            # Expected Calibration Error (ECE)
            bin_counts = np.histogram(y_pred[:, i], bins=n_bins, range=(0, 1))[0]
            bin_weights = bin_counts / len(y_pred)
            
            ece = np.sum(bin_weights[:len(prob_true)] * np.abs(prob_true - prob_pred))
            ece_scores.append(ece)
            
            # Maximum Calibration Error (MCE)
            mce = np.max(np.abs(prob_true - prob_pred))
            mce_scores.append(mce)
            
        except Exception as e:
            warnings.warn(f"Could not compute calibration for label {i}: {e}")
    
    metrics = {}
    if ece_scores:
        metrics['mean_ece'] = np.mean(ece_scores)
        metrics['mean_mce'] = np.mean(mce_scores)
    
    return metrics


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Focusing parameter for down-weighting easy examples
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch_size, num_labels)
            targets: Ground truth labels (batch_size, num_labels)
        """
        # Compute BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Compute probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute focal term: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute alpha term
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Addresses the issue of negative-positive imbalance in multi-label datasets.
    
    Args:
        gamma_neg: Focusing parameter for negative examples
        gamma_pos: Focusing parameter for positive examples
        clip: Clipping value for probabilities (prevents log(0))
    """
    
    def __init__(
        self, 
        gamma_neg: float = 4.0, 
        gamma_pos: float = 1.0, 
        clip: float = 0.05
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch_size, num_labels)
            targets: Ground truth labels (batch_size, num_labels)
        """
        # Compute probabilities
        probs = torch.sigmoid(inputs)
        
        # Asymmetric clipping
        probs_pos = torch.clamp(probs, min=self.clip)
        probs_neg = torch.clamp(1 - probs, min=self.clip)
        
        # Positive loss
        loss_pos = targets * torch.log(probs_pos)
        loss_pos = loss_pos * (1 - probs) ** self.gamma_pos
        
        # Negative loss
        loss_neg = (1 - targets) * torch.log(probs_neg)
        loss_neg = loss_neg * probs ** self.gamma_neg
        
        # Combine
        loss = -(loss_pos + loss_neg)
        
        return loss.mean()


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for minimization, 'max' for maximization
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str
):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: str = 'cpu'
) -> Tuple[int, Dict[str, float]]:
    """
    Load model checkpoint.
    
    Returns:
        epoch: Epoch number
        metrics: Saved metrics
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    
    return epoch, metrics


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_pos_weights(train_labels: np.ndarray) -> torch.Tensor:
    """
    Compute positive class weights for handling class imbalance.
    
    Args:
        train_labels: Training labels (n_samples, n_labels)
        
    Returns:
        Tensor of positive weights for each label
    """
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    
    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)
    
    # Weight = neg_count / pos_count
    pos_weights = neg_counts / pos_counts
    
    return torch.FloatTensor(pos_weights)
