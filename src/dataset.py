"""
Dataset loader for EchoNext data with ECG waveforms and tabular features.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Tuple, Optional


class EchoNextDataset(Dataset):
    """
    Dataset for loading EchoNext ECG waveforms and tabular features.
    
    Args:
        waveform_path: Path to .npy file containing ECG waveforms (N x 1 x 2500 x 12)
        tabular_path: Path to .npy file containing tabular features (N x 7)
        metadata_path: Path to metadata CSV file
        split: One of 'train', 'val', 'test', or 'no_split'
    """
    
    # Label columns (excluding composite SHD flag which we'll handle separately)
    LABEL_COLUMNS = [
        'lvef_lte_45_flag',
        'lvwt_gte_13_flag',
        'aortic_stenosis_moderate_or_greater_flag',
        'aortic_regurgitation_moderate_or_greater_flag',
        'mitral_regurgitation_moderate_or_greater_flag',
        'tricuspid_regurgitation_moderate_or_greater_flag',
        'pulmonary_regurgitation_moderate_or_greater_flag',
        'rv_systolic_dysfunction_moderate_or_greater_flag',
        'pericardial_effusion_moderate_large_flag',
        'pasp_gte_45_flag',
        'tr_max_gte_32_flag',
        'shd_moderate_or_greater_flag'  # Composite label
    ]
    
    # Tabular feature names in order
    TABULAR_FEATURES = [
        'sex',
        'ventricular_rate',
        'atrial_rate',
        'pr_interval',
        'qrs_duration',
        'qt_corrected',
        'age_at_ecg'
    ]
    
    def __init__(
        self,
        waveform_path: str,
        tabular_path: str,
        metadata_path: str,
        split: str
    ):
        super().__init__()
        
        # Load waveforms and tabular features
        self.waveforms = np.load(waveform_path)
        self.tabular = np.load(tabular_path)
        
        # Print actual shapes for debugging
        print(f"Loaded waveforms with shape: {self.waveforms.shape}")
        print(f"Loaded tabular with shape: {self.tabular.shape}")
        
        # Load metadata and filter by split
        metadata = pd.read_csv(metadata_path)
        self.metadata = metadata[metadata['split'] == split].reset_index(drop=True)
        
        # Extract labels (handle missing values by filling with 0)
        self.labels = self.metadata[self.LABEL_COLUMNS].fillna(0).values.astype(np.float32)
        
        # Store original tabular data from metadata for missingness detection
        # Convert to numeric, coercing errors to NaN for proper missingness detection
        self.tabular_raw = self.metadata[self.TABULAR_FEATURES].apply(
            pd.to_numeric, errors='coerce'
        ).values.astype(np.float32)
        
        # Validate shapes
        assert self.waveforms.shape[0] == len(self.metadata), \
            f"Waveform count {self.waveforms.shape[0]} != metadata count {len(self.metadata)}"
        assert self.tabular.shape[0] == len(self.metadata), \
            f"Tabular count {self.tabular.shape[0]} != metadata count {len(self.metadata)}"
        
        print(f"Loaded {split} split: {len(self)} samples")
        print(f"  Waveform shape: {self.waveforms.shape}")
        print(f"  Tabular shape: {self.tabular.shape}")
        print(f"  Labels shape: {self.labels.shape}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing:
        - waveform: ECG waveform (1, 2500, 12)
        - tabular: Preprocessed tabular features (7,)
        - tabular_mask: Binary mask indicating missing values (7,)
        - labels: Multi-label binary targets (12,)
        """
        # Get waveform
        waveform = torch.from_numpy(self.waveforms[idx]).float()  # (1, 2500, 12)
        
        # Get preprocessed tabular features
        tabular = torch.from_numpy(self.tabular[idx]).float()  # (7,)
        
        # Create missingness mask from raw metadata
        # Missing values in the raw data are NaN or special sentinel values
        tabular_raw = self.tabular_raw[idx]
        tabular_mask = torch.from_numpy(~np.isnan(tabular_raw)).float()  # 1 = present, 0 = missing
        
        # Get labels
        labels = torch.from_numpy(self.labels[idx]).float()  # (12,)
        
        return {
            'waveform': waveform,
            'tabular': tabular,
            'tabular_mask': tabular_mask,
            'labels': labels,
            'idx': idx
        }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create dataloaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing the echonext_dataset folder
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader, test_loader
    """
    import os
    
    metadata_path = os.path.join(data_dir, 'EchoNext_metadata_100k.csv')
    
    # Create datasets
    train_dataset = EchoNextDataset(
        waveform_path=os.path.join(data_dir, 'EchoNext_train_waveforms.npy'),
        tabular_path=os.path.join(data_dir, 'EchoNext_train_tabular_features.npy'),
        metadata_path=metadata_path,
        split='train'
    )
    
    val_dataset = EchoNextDataset(
        waveform_path=os.path.join(data_dir, 'EchoNext_val_waveforms.npy'),
        tabular_path=os.path.join(data_dir, 'EchoNext_val_tabular_features.npy'),
        metadata_path=metadata_path,
        split='val'
    )
    
    test_dataset = EchoNextDataset(
        waveform_path=os.path.join(data_dir, 'EchoNext_test_waveforms.npy'),
        tabular_path=os.path.join(data_dir, 'EchoNext_test_tabular_features.npy'),
        metadata_path=metadata_path,
        split='test'
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
