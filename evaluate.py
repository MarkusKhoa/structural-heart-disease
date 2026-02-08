"""
Evaluation and inference script for the SHD Multimodal Model.

This script can be used to:
1. Evaluate a trained model on test data
2. Perform inference on new data
3. Generate predictions with uncertainty estimates
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.models import SHDMultimodalModel
from src.dataset import EchoNextDataset
from src.utils import compute_metrics, compute_calibration_metrics, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SHD Multimodal Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./echonext_dataset',
                        help='Path to EchoNext dataset directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--uncertainty_samples', type=int, default=20,
                        help='Number of MC dropout samples for uncertainty')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save individual predictions to CSV')
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_uncertainty_samples=20, save_individual=False):
    """
    Evaluate model and collect predictions.
    
    Returns:
        metrics: Dictionary of evaluation metrics
        results_df: DataFrame with predictions (if save_individual=True)
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    all_calibrated_probs = []
    all_uncertainties = []
    all_fusion_gates = []
    all_indices = []
    
    print("Running evaluation...")
    for batch in tqdm(dataloader):
        # Move data to device
        waveform = batch['waveform'].to(device)
        tabular = batch['tabular'].to(device)
        tabular_mask = batch['tabular_mask'].to(device)
        labels = batch['labels'].to(device)
        indices = batch['idx']
        
        # Standard forward pass
        output = model(waveform, tabular, tabular_mask)
        
        # Uncertainty estimation
        uncertainty_output = model.predict_with_uncertainty(
            waveform, tabular, tabular_mask, num_samples=num_uncertainty_samples
        )
        
        # Store results
        all_labels.append(labels.cpu().numpy())
        all_probs.append(output['probs'].cpu().numpy())
        all_calibrated_probs.append(output['calibrated_probs'].cpu().numpy())
        all_uncertainties.append(uncertainty_output['uncertainty'].cpu().numpy())
        all_fusion_gates.append(output['fusion_gates'].cpu().numpy())
        all_indices.extend(indices.tolist())
    
    # Concatenate all results
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_calibrated_probs = np.concatenate(all_calibrated_probs, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_fusion_gates = np.concatenate(all_fusion_gates, axis=0)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(
        all_labels, all_calibrated_probs, EchoNextDataset.LABEL_COLUMNS
    )
    calibration_metrics = compute_calibration_metrics(all_labels, all_calibrated_probs)
    metrics.update(calibration_metrics)
    
    # Add fusion gate statistics
    metrics['avg_ecg_gate'] = float(all_fusion_gates[:, 0].mean())
    metrics['avg_tabular_gate'] = float(all_fusion_gates[:, 1].mean())
    metrics['avg_uncertainty'] = float(all_uncertainties.mean())
    
    # Create results DataFrame if requested
    results_df = None
    if save_individual:
        print("\nCreating predictions DataFrame...")
        results_data = {
            'sample_idx': all_indices,
            'ecg_gate': all_fusion_gates[:, 0],
            'tabular_gate': all_fusion_gates[:, 1],
        }
        
        # Add predictions and labels for each condition
        for i, label_name in enumerate(EchoNextDataset.LABEL_COLUMNS):
            results_data[f'{label_name}_true'] = all_labels[:, i]
            results_data[f'{label_name}_prob'] = all_probs[:, i]
            results_data[f'{label_name}_calibrated_prob'] = all_calibrated_probs[:, i]
            results_data[f'{label_name}_uncertainty'] = all_uncertainties[:, i]
        
        results_df = pd.DataFrame(results_data)
    
    return metrics, results_df


def print_metrics(metrics, title="Evaluation Results"):
    """Pretty print evaluation metrics."""
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    
    # Overall metrics
    print("\n📊 Overall Performance:")
    print(f"  Macro AUROC:      {metrics.get('macro_auroc', 0):.4f}")
    print(f"  Macro AUPRC:      {metrics.get('macro_auprc', 0):.4f}")
    
    # Calibration metrics
    print("\n🎯 Calibration:")
    print(f"  Expected Cal Err: {metrics.get('mean_ece', 0):.4f}")
    print(f"  Maximum Cal Err:  {metrics.get('mean_mce', 0):.4f}")
    
    # Uncertainty
    print("\n🔮 Uncertainty:")
    print(f"  Avg Uncertainty:  {metrics.get('avg_uncertainty', 0):.4f}")
    
    # Fusion gates
    print("\n🔀 Modality Fusion:")
    print(f"  ECG Weight:       {metrics.get('avg_ecg_gate', 0):.3f}")
    print(f"  Tabular Weight:   {metrics.get('avg_tabular_gate', 0):.3f}")
    
    # Per-label AUROC
    print("\n📋 Per-Label AUROC:")
    for label in EchoNextDataset.LABEL_COLUMNS:
        auroc = metrics.get(f'{label}_auroc', None)
        if auroc is not None:
            status = "✓" if auroc >= 0.75 else "⚠"
            print(f"  {status} {label:<50s} {auroc:.4f}")
    
    print("="*60 + "\n")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    metadata_path = f"{args.data_dir}/EchoNext_metadata_100k.csv"
    waveform_path = f"{args.data_dir}/EchoNext_{args.split}_waveforms.npy"
    tabular_path = f"{args.data_dir}/EchoNext_{args.split}_tabular_features.npy"
    
    dataset = EchoNextDataset(
        waveform_path=waveform_path,
        tabular_path=tabular_path,
        metadata_path=metadata_path,
        split=args.split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = SHDMultimodalModel()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, None, args.checkpoint, device)
    model = model.to(device)
    
    # Evaluate
    metrics, results_df = evaluate_model(
        model, dataloader, device,
        num_uncertainty_samples=args.uncertainty_samples,
        save_individual=args.save_predictions
    )
    
    # Print results
    print_metrics(metrics, f"{args.split.upper()} Set Evaluation Results")
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    
    # Save metrics
    with open(output_dir / f'{args.split}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions if requested
    if results_df is not None:
        results_df.to_csv(output_dir / f'{args.split}_predictions.csv', index=False)
        print(f"  ✓ Saved predictions to {args.split}_predictions.csv")
    
    print(f"  ✓ Saved metrics to {args.split}_metrics.json")
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
