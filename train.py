"""
Training script for Structural Heart Disease multimodal prediction model.

This script trains the complete A1-A3 pipeline:
- A1: ECG Transformer Encoder
- A2: Tabular Encoder with missingness modeling
- A3: Multimodal fusion + Multi-label prediction

Outputs include per-label probabilities, calibration, and uncertainty.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.models import SHDMultimodalModel
from src.dataset import get_dataloaders, EchoNextDataset
from src.utils import (
    compute_metrics, compute_calibration_metrics,
    FocalLoss, AsymmetricLoss, EarlyStopping,
    save_checkpoint, load_checkpoint, AverageMeter,
    get_pos_weights
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SHD Multimodal Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./echonext_dataset',
                        help='Path to EchoNext dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--ecg_model_size', type=str, default='large',
                        choices=['small', 'base', 'large'],
                        help='Size of pretrained HuBERT-ECG model')
    parser.add_argument('--ecg_embed_dim', type=int, default=256,
                        help='ECG embedding dimension')
    parser.add_argument('--ecg_freeze', action='store_true',
                        help='Freeze pretrained HuBERT-ECG encoder')
    parser.add_argument('--ecg_use_pretrained', action='store_true', default=True,
                        help='Use pretrained HuBERT-ECG weights')
    parser.add_argument('--tabular_dim', type=int, default=32,
                        help='Tabular feature embedding dimension')
    parser.add_argument('--tabular_depth', type=int, default=2,
                        help='Number of transformer layers in tabular encoder')
    parser.add_argument('--tabular_heads', type=int, default=4,
                        help='Number of attention heads in tabular encoder')
    parser.add_argument('--tabular_output_dim', type=int, default=128,
                        help='Tabular final output dimension')
    parser.add_argument('--fusion_dim', type=int, default=256,
                        help='Fusion output dimension')
    parser.add_argument('--fusion_dropout', type=float, default=0.1,
                        help='Dropout rate in fusion gating network')
    parser.add_argument('--fusion_cross_interaction', action='store_true', default=True,
                        help='Enable cross-modal SE-style interaction in fusion')
    parser.add_argument('--no_fusion_cross_interaction', action='store_true',
                        help='Disable cross-modal interaction in fusion')
    parser.add_argument('--gate_balance_weight', type=float, default=0.01,
                        help='Weight for auxiliary gate regularisation loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='asymmetric',
                        choices=['bce', 'focal', 'asymmetric'],
                        help='Type of loss function')
    parser.add_argument('--use_pos_weights', action='store_true',
                        help='Use positive class weights for imbalanced data')
    
    # Regularization arguments
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Computational arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda/cpu)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Evaluation arguments
    parser.add_argument('--eval_uncertainty', action='store_true',
                        help='Evaluate with uncertainty estimation')
    parser.add_argument('--uncertainty_samples', type=int, default=20,
                        help='Number of MC dropout samples for uncertainty')
    
    return parser.parse_args()


def get_loss_function(args, pos_weights=None):
    """Get loss function based on arguments."""
    if args.loss_type == 'bce':
        if args.use_pos_weights and pos_weights is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif args.loss_type == 'asymmetric':
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    return criterion


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler=None):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move data to device
        waveform = batch['waveform'].to(device)
        tabular = batch['tabular'].to(device)
        tabular_mask = batch['tabular_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(waveform, tabular, tabular_mask)
                loss = criterion(output['logits'], labels)
                # Add gated-fusion auxiliary regularisation loss
                if 'auxiliary_loss' in output:
                    loss = loss + output['auxiliary_loss']
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(waveform, tabular, tabular_mask)
            loss = criterion(output['logits'], labels)
            # Add gated-fusion auxiliary regularisation loss
            if 'auxiliary_loss' in output:
                loss = loss + output['auxiliary_loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        losses.update(loss.item(), waveform.size(0))
        
        # Update progress bar
        pbar.set_postfix({'loss': losses.avg})
    
    return losses.avg


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, eval_uncertainty=False, num_samples=20):
    """Evaluate model."""
    model.eval()
    
    losses = AverageMeter()
    all_labels = []
    all_probs = []
    all_calibrated_probs = []
    all_uncertainties = []
    all_fusion_gates = []
    
    pbar = tqdm(dataloader, desc='Evaluating')
    for batch in pbar:
        # Move data to device
        waveform = batch['waveform'].to(device)
        tabular = batch['tabular'].to(device)
        tabular_mask = batch['tabular_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        output = model(waveform, tabular, tabular_mask)
        loss = criterion(output['logits'], labels)
        
        # Update loss
        losses.update(loss.item(), waveform.size(0))
        
        # Store predictions and labels
        all_labels.append(labels.cpu().numpy())
        all_probs.append(output['probs'].cpu().numpy())
        all_calibrated_probs.append(output['calibrated_probs'].cpu().numpy())
        all_fusion_gates.append(output['fusion_gates'].cpu().numpy())
        
        # Uncertainty estimation if requested
        if eval_uncertainty:
            uncertainty_output = model.predict_with_uncertainty(
                waveform, tabular, tabular_mask, num_samples=num_samples
            )
            all_uncertainties.append(uncertainty_output['uncertainty'].cpu().numpy())
    
    # Concatenate all batches
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_calibrated_probs = np.concatenate(all_calibrated_probs, axis=0)
    all_fusion_gates = np.concatenate(all_fusion_gates, axis=0)
    
    if eval_uncertainty:
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_calibrated_probs, EchoNextDataset.LABEL_COLUMNS)
    calibration_metrics = compute_calibration_metrics(all_labels, all_calibrated_probs)
    
    metrics.update(calibration_metrics)
    metrics['loss'] = losses.avg
    
    # Compute average fusion gate values
    metrics['avg_ecg_gate'] = all_fusion_gates[:, 0].mean()
    metrics['avg_tabular_gate'] = all_fusion_gates[:, 1].mean()
    
    if eval_uncertainty:
        metrics['avg_uncertainty'] = all_uncertainties.mean()
    
    return metrics, all_probs, all_calibrated_probs, all_uncertainties if eval_uncertainty else None


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Compute positive weights for imbalanced data
    pos_weights = None
    if args.use_pos_weights:
        # Get training labels
        train_dataset = train_loader.dataset
        train_labels = train_dataset.labels
        pos_weights = get_pos_weights(train_labels).to(device)
        print(f"Using positive weights: {pos_weights}")
    
    # Create model
    print("Creating model...")
    model_config = {
        'ecg_config': {
            'model_size': args.ecg_model_size,
            'embed_dim': args.ecg_embed_dim,
            'freeze_encoder': args.ecg_freeze,
            'use_pretrained': args.ecg_use_pretrained,
            'pooling': 'mean',
        },
        'tabular_config': {
            'dim': args.tabular_dim,
            'depth': args.tabular_depth,
            'heads': args.tabular_heads,
            'output_dim': args.tabular_output_dim,
            'attn_dropout': args.dropout,
            'ff_dropout': args.dropout,
        },
        'fusion_config': {
            'ecg_dim': args.ecg_embed_dim,
            'tabular_dim': args.tabular_output_dim,
            'output_dim': args.fusion_dim,
            'dropout': args.fusion_dropout,
            'use_cross_interaction': args.fusion_cross_interaction and not args.no_fusion_cross_interaction,
            'gate_balance_weight': args.gate_balance_weight,
        },
        'prediction_config': {
            'input_dim': args.fusion_dim,
            'num_labels': len(EchoNextDataset.LABEL_COLUMNS),
            'dropout': args.dropout,
        }
    }
    
    model = SHDMultimodalModel(**model_config)
    model = model.to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Create loss function
    criterion = get_loss_function(args, pos_weights)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode='max'  # Maximize validation AUROC
    )
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
    
    # Training loop
    print("\nStarting training...")
    best_val_auroc = 0.0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        
        # Evaluate on validation set
        val_metrics, _, _, _ = evaluate(
            model, val_loader, criterion, device,
            eval_uncertainty=args.eval_uncertainty,
            num_samples=args.uncertainty_samples
        )
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('AUROC/val_macro', val_metrics.get('macro_auroc', 0), epoch)
        writer.add_scalar('AUPRC/val_macro', val_metrics.get('macro_auprc', 0), epoch)
        writer.add_scalar('Calibration/val_ece', val_metrics.get('mean_ece', 0), epoch)
        writer.add_scalar('FusionGates/ecg', val_metrics['avg_ecg_gate'], epoch)
        writer.add_scalar('FusionGates/tabular', val_metrics['avg_tabular_gate'], epoch)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Macro AUROC: {val_metrics.get('macro_auroc', 0):.4f}")
        print(f"Val Macro AUPRC: {val_metrics.get('macro_auprc', 0):.4f}")
        print(f"Val ECE: {val_metrics.get('mean_ece', 0):.4f}")
        print(f"Fusion Gates - ECG: {val_metrics['avg_ecg_gate']:.3f}, Tabular: {val_metrics['avg_tabular_gate']:.3f}")
        
        # Save best model
        val_auroc = val_metrics.get('macro_auroc', 0)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / 'best_model.pt'
            )
            print(f"✓ New best model saved (AUROC: {val_auroc:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            )
        
        # Early stopping
        if early_stopping(val_auroc):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    load_checkpoint(model, None, output_dir / 'best_model.pt', device)
    
    test_metrics, test_probs, test_calibrated_probs, test_uncertainties = evaluate(
        model, test_loader, criterion, device,
        eval_uncertainty=True,
        num_samples=args.uncertainty_samples
    )
    
    # Print test results
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Macro AUROC: {test_metrics.get('macro_auroc', 0):.4f}")
    print(f"Test Macro AUPRC: {test_metrics.get('macro_auprc', 0):.4f}")
    print(f"Test ECE: {test_metrics.get('mean_ece', 0):.4f}")
    print(f"Test MCE: {test_metrics.get('mean_mce', 0):.4f}")
    print(f"Average Uncertainty: {test_metrics.get('avg_uncertainty', 0):.4f}")
    print("\nPer-label AUROC:")
    for label in EchoNextDataset.LABEL_COLUMNS:
        auroc = test_metrics.get(f'{label}_auroc', None)
        if auroc is not None:
            print(f"  {label}: {auroc:.4f}")
    
    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save predictions
    np.save(output_dir / 'test_probs.npy', test_probs)
    np.save(output_dir / 'test_calibrated_probs.npy', test_calibrated_probs)
    if test_uncertainties is not None:
        np.save(output_dir / 'test_uncertainties.npy', test_uncertainties)
    
    print(f"\nTraining complete! Results saved to {output_dir}")
    writer.close()


if __name__ == '__main__':
    main()
