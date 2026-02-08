"""
Core model components: ECG Encoder, Tabular Encoder, and Multimodal Fusion.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from transformers import AutoModel
import warnings


class ECGTransformerEncoder(nn.Module):
    """
    A1: ECG Encoder using Pretrained HuBERT-ECG model.
    
    Uses the pretrained HuBERT-ECG foundation model from Edoardo-BS.
    The model is pre-trained on 9.1M 12-lead ECGs for self-supervised learning.
    
    Args:
        model_size: Size of HuBERT model ('small', 'base', 'large') (default: 'large')
        embed_dim: Output embedding dimension (default: 256)
        freeze_encoder: Whether to freeze the pretrained encoder (default: False)
        pooling: Pooling method ('mean', 'max', 'cls') (default: 'mean')
        use_pretrained: Whether to use pretrained weights (default: True)
    """
    
    def __init__(
        self,
        model_size: str = 'large',
        embed_dim: int = 256,
        freeze_encoder: bool = False,
        pooling: str = 'mean',
        use_pretrained: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pooling = pooling
        self.model_size = model_size
        
        # Load pretrained HuBERT-ECG model
        if use_pretrained:
            try:
                print(f"Loading pretrained HuBERT-ECG-{model_size} model...")
                self.hubert_ecg = AutoModel.from_pretrained(
                    f"Edoardo-BS/hubert-ecg-{model_size}",
                    trust_remote_code=True
                )
                print(f"✓ Loaded pretrained HuBERT-ECG-{model_size}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load pretrained HuBERT-ECG: {e}. "
                    f"Using random initialization instead."
                )
                # Fallback to random initialization if pretrained model fails
                self.hubert_ecg = self._create_fallback_model()
        else:
            print("Initializing HuBERT-ECG with random weights...")
            self.hubert_ecg = self._create_fallback_model()
        
        # Freeze encoder if requested (for transfer learning)
        if freeze_encoder:
            for param in self.hubert_ecg.parameters():
                param.requires_grad = False
            print("✓ Froze HuBERT-ECG encoder weights")
        
        # Get the output dimension of HuBERT
        # Small: 768, Base: 768, Large: 1024
        hubert_dim = self._get_hubert_output_dim()
        
        # Projection layer to match desired embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(hubert_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
    def _get_hubert_output_dim(self) -> int:
        """Get the output dimension of the HuBERT model."""
        # Check if we have a loaded model with config
        if hasattr(self, 'hubert_ecg') and hasattr(self.hubert_ecg, 'config'):
            return self.hubert_ecg.config.hidden_size
        
        # Fallback to default dimensions
        dim_map = {
            'small': 768,
            'base': 768,
            'large': 960  # Updated based on actual HuBERT-ECG-large config
        }
        return dim_map.get(self.model_size, 768)
    
    def _create_fallback_model(self):
        """Create a simple fallback model if pretrained weights fail to load."""
        # Simple CNN + Transformer fallback
        class FallbackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(12, 128, kernel_size=15, stride=4, padding=7),
                    nn.BatchNorm1d(128),
                    nn.GELU(),
                    nn.Conv1d(128, 256, kernel_size=15, stride=4, padding=7),
                    nn.BatchNorm1d(256),
                    nn.GELU(),
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=256, nhead=8, dim_feedforward=1024,
                    dropout=0.1, activation='gelu', batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
                
            def forward(self, x):
                # x shape: (B, 12, 2500)
                x = self.conv(x)  # (B, 256, reduced_len)
                x = x.permute(0, 2, 1)  # (B, seq_len, 256)
                x = self.transformer(x)
                return x
        
        return FallbackModel()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input waveform (batch_size, 1, channels, seq_length) OR (batch_size, 1, seq_length, channels)
               Expected: (B, 1, 12, 2500) or (B, 1, 2500, 12) - 10 seconds at 250Hz
            
        Returns:
            embedding: Pooled global embedding (batch_size, embed_dim)
            sequence: Sequence of embeddings (batch_size, seq_len, hubert_dim)
        """
        # Reshape input: (B, 1, 2500, 12) or (B, 1, 12, 2500) -> (B, 12*2500)
        # HuBERT-ECG expects (batch, seq_length) for single-channel audio-like input
        if x.dim() == 4:
            batch_size = x.shape[0]
            # Remove singleton dimension and flatten 12 leads
            if x.shape[1] == 1 and x.shape[2] == 2500 and x.shape[3] == 12:
                # (B, 1, 2500, 12) -> (B, 30000)
                x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size, -1)
            elif x.shape[1] == 1 and x.shape[2] == 12 and x.shape[3] == 2500:
                # (B, 1, 12, 2500) -> (B, 30000)
                x = x.squeeze(1).reshape(batch_size, -1)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Pass through HuBERT-ECG
        # HuBERT expects (batch, 1, seq_length) for single-channel input
        output = self.hubert_ecg(x)
        sequence = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output
        
        # Pooling over sequence dimension
        if self.pooling == 'mean':
            pooled = sequence.mean(dim=1)  # (B, hubert_dim)
        elif self.pooling == 'max':
            pooled = sequence.max(dim=1)[0]  # (B, hubert_dim)
        elif self.pooling == 'cls':
            pooled = sequence[:, 0]  # Take first token (B, hubert_dim)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Project to desired embedding dimension
        embedding = self.projection(pooled)  # (B, embed_dim)
        
        return embedding, sequence


class TabularEncoder(nn.Module):
    """
    A2: Tabular Encoder using FTTransformer from the official implementation.
    
    Uses Feature Tokenizer Transformer (FTTransformer) which is better suited
    for continuous/numerical tabular features compared to the original TabTransformer.
    
    FTTransformer treats each continuous feature as a token with learned embeddings,
    applies self-attention across features, and handles missing values explicitly.
    
    Paper: "Revisiting Deep Learning Models for Tabular Data" (Yandex Research)
    Implementation: lucidrains/tab-transformer-pytorch
    
    Args:
        input_dim: Number of input features (default: 7)
        dim: Feature embedding dimension (default: 32)
        depth: Number of transformer layers (default: 2)
        heads: Number of attention heads (default: 4)
        attn_dropout: Attention dropout rate (default: 0.1)
        ff_dropout: Feed-forward dropout rate (default: 0.1)
        output_dim: Final output embedding dimension (default: 128)
        use_official_impl: Whether to use official FTTransformer (default: True)
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        dim: int = 32,
        depth: int = 2,
        heads: int = 4,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        output_dim: int = 128,
        use_official_impl: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.dim = dim
        self.output_dim = output_dim
        # Alias for consistency with other encoders (e.g., ECGTransformerEncoder)
        # so downstream code can refer to `.embed_dim` uniformly.
        self.embed_dim = output_dim
        self.use_official_impl = use_official_impl
        
        if use_official_impl:
            try:
                from tab_transformer_pytorch import FTTransformer
                
                print("Using official FTTransformer implementation...")
                # FTTransformer expects no categories (all continuous)
                # It handles continuous features with feature-wise embeddings
                self.ft_transformer = FTTransformer(
                    categories=(),  # No categorical features
                    num_continuous=input_dim,  # All 7 features are continuous
                    dim=dim,
                    dim_out=output_dim,  # Output dimension
                    depth=depth,
                    heads=heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout
                )
                print("✓ Loaded official FTTransformer")
                
            except ImportError as e:
                warnings.warn(
                    f"Failed to import FTTransformer: {e}. "
                    f"Install with: pip install tab-transformer-pytorch. "
                    f"Using fallback implementation."
                )
                self.use_official_impl = False
                self._create_fallback_model(dim, depth, heads, attn_dropout, ff_dropout)
        else:
            print("Using fallback TabTransformer implementation...")
            self._create_fallback_model(dim, depth, heads, attn_dropout, ff_dropout)
    
    def _create_fallback_model(self, dim, depth, heads, attn_dropout, ff_dropout):
        """Create fallback model if official implementation not available."""
        # Per-feature embeddings
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, dim),
                nn.LayerNorm(dim)
            ) for _ in range(self.input_dim)
        ])
        
        # Positional embeddings
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, self.input_dim, dim) * 0.02
        )
        
        # Missingness embeddings
        self.missing_embeddings = nn.Parameter(
            torch.randn(self.input_dim, dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=ff_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.input_dim * dim, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )
        
    def forward(
        self, 
        features: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: Preprocessed tabular features (batch_size, input_dim)
            mask: Binary mask (1=present, 0=missing) (batch_size, input_dim)
            
        Returns:
            embedding: Tabular embedding (batch_size, output_dim)
            info: Dictionary with missingness information for interpretability
        """
        batch_size = features.size(0)
        
        # Handle missing values: replace with 0 for masked positions
        # FTTransformer can handle this in its continuous feature processing
        features_masked = features * mask
        
        if self.use_official_impl:
            # Official FTTransformer expects:
            # - x_categ: categorical features (we have none, so empty tensor)
            # - x_numer: numerical/continuous features
            x_categ = torch.empty(batch_size, 0, dtype=torch.long, device=features.device)
            
            # Pass through FTTransformer
            embedding = self.ft_transformer(x_categ, features_masked)
            
        else:
            # Fallback implementation
            feature_embeds = []
            for i in range(self.input_dim):
                feat_val = features_masked[:, i:i+1]
                is_present = mask[:, i:i+1]
                
                # Embed feature
                feat_embed = self.feature_embeddings[i](feat_val)
                
                # Use missingness embedding for missing values
                missing_embed = self.missing_embeddings[i].unsqueeze(0).expand(batch_size, -1)
                feat_embed = is_present * feat_embed + (1 - is_present) * missing_embed
                
                feature_embeds.append(feat_embed)
            
            # Stack and process through transformer
            x = torch.stack(feature_embeds, dim=1)
            x = x + self.pos_embeddings
            x = self.transformer(x)
            x = x.reshape(batch_size, -1)
            embedding = self.output_projection(x)
        
        # Store missingness information
        info = {
            'missingness_rate': 1.0 - mask.mean(dim=0),
            'sample_missingness': 1.0 - mask.mean(dim=1),
            'mask': mask
        }
        
        return embedding, info


class GatedFusion(nn.Module):
    """
    A3: Gated Multimodal Fusion with element-wise gating.

    Implements feature-level gated fusion following the Gated Multimodal Unit
    (GMU) framework (Arevalo et al., 2017) with extensions for cross-modal
    interaction and auxiliary regularization.

    Unlike scalar gating (one weight per modality), this module learns
    **element-wise** gates: each dimension of the fused embedding independently
    decides how much to draw from the ECG vs. tabular representation.  This
    lets the network selectively combine complementary information -- e.g.,
    rhythm features from the ECG with age/sex from the tabular side -- at a
    much finer granularity.

    Architecture overview::

        ┌─────────────┐           ┌──────────────────┐
        │ ECG encoder  │──ecg_emb──►  Tanh-projection  │──h_ecg──┐
        └─────────────┘           └──────────────────┘          │
                                                                 ▼
                                              ┌──────────────────────────────┐
                                              │  Cross-modal interaction     │
                                              │  (optional SE-style gating)  │
                                              └──────────────────────────────┘
                                                                 │
        ┌──────────────┐          ┌──────────────────┐          │
        │ Tab. encoder  │──tab_emb─►  Tanh-projection  │──h_tab──┘
        └──────────────┘          └──────────────────┘
                │                                        │
                └────────cat(ecg_emb, tab_emb)───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Gating network (MLP)  │
                    │   → element-wise σ      │
                    └────────────┬────────────┘
                                 │ gates ∈ [0,1]^d
                                 ▼
              fused = gates ⊙ h_ecg + (1 − gates) ⊙ h_tab
                                 │
                          LayerNorm + Dropout

    Auxiliary losses (added to the main task loss):
        * **Balance loss** -- penalises the batch-mean gate for drifting away
          from 0.5, encouraging both modalities to contribute.
        * **Entropy loss** -- penalises gate values near 0.5, encouraging each
          dimension to make a decisive choice.

    The two losses are complementary: balance prevents *global* collapse to
    one modality, while entropy encourages *local* (per-dimension) decisiveness.

    Args:
        ecg_dim: ECG embedding dimension (default: 256)
        tabular_dim: Tabular embedding dimension (default: 128)
        hidden_dim: Hidden dimension for the gating network (default: 256)
        output_dim: Output fused embedding dimension (default: 256)
        dropout: Dropout rate for gating network and output (default: 0.1)
        use_cross_interaction: Enable cross-modal SE-style interaction
            before gating (default: True)
        gate_balance_weight: Scalar weight for the auxiliary loss term
            that is added to the main task loss (default: 0.01)
    """

    def __init__(
        self,
        ecg_dim: int = 256,
        tabular_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
        use_cross_interaction: bool = True,
        gate_balance_weight: float = 0.01,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.use_cross_interaction = use_cross_interaction
        self.gate_balance_weight = gate_balance_weight

        # ── Modality projection (GMU-style with tanh) ────────────────────
        # Tanh bounds representations to [-1, 1], stabilising the gated sum.
        self.ecg_transform = nn.Sequential(
            nn.Linear(ecg_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )
        self.tabular_transform = nn.Sequential(
            nn.Linear(tabular_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

        # ── Cross-modal interaction (SE-style gated modulation) ──────────
        # Each modality is multiplicatively modulated by a gate computed
        # from the *joint* representation, allowing mutual enrichment.
        if use_cross_interaction:
            self.cross_ecg_gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid(),
            )
            self.cross_tabular_gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid(),
            )

        # ── Element-wise gating network ──────────────────────────────────
        # Operates on the *raw* (pre-transform) embeddings so the gate
        # decision is based on original modality signals, not the fused space.
        gate_input_dim = ecg_dim + tabular_dim
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # element-wise gates in [0, 1]
        )

        # ── Output normalisation & regularisation ────────────────────────
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)

    # ── helpers ───────────────────────────────────────────────────────────

    def _cross_interact(
        self,
        h_ecg: torch.Tensor,
        h_tabular: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SE-style cross-modal gated modulation.

        Each modality is element-wise scaled by a sigmoid gate derived from
        the concatenation of *both* transformed modality representations.

        Args:
            h_ecg: Transformed ECG embedding ``(B, output_dim)``
            h_tabular: Transformed tabular embedding ``(B, output_dim)``

        Returns:
            Modulated ``(h_ecg, h_tabular)`` pair.
        """
        combined = torch.cat([h_ecg, h_tabular], dim=-1)  # (B, 2 * output_dim)

        ecg_scale = self.cross_ecg_gate(combined)           # (B, output_dim)
        tabular_scale = self.cross_tabular_gate(combined)   # (B, output_dim)

        h_ecg = h_ecg * ecg_scale
        h_tabular = h_tabular * tabular_scale

        return h_ecg, h_tabular

    def compute_auxiliary_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """Compute gate regularisation loss.

        Combines two complementary objectives:

        1. **Balance loss** -- ``mean_over_dims((E_batch[g] - 0.5)^2)``
           penalises the batch-average gate for each dimension if it strays
           from 0.5, preventing global collapse to a single modality.

        2. **Entropy loss** -- ``mean(-g log g - (1-g) log(1-g))``
           penalises gates sitting at 0.5 (maximum entropy), encouraging
           each dimension to make a decisive binary choice.

        Args:
            gates: Element-wise gate values ``(B, output_dim)`` in [0, 1].

        Returns:
            Scalar auxiliary loss (already weighted by ``gate_balance_weight``).
        """
        eps = 1e-7

        # Balance: batch-mean gate per dimension should be near 0.5
        mean_gate = gates.mean(dim=0)  # (output_dim,)
        balance_loss = ((mean_gate - 0.5) ** 2).mean()

        # Entropy: per-element gates should be decisive (near 0 or 1)
        entropy = -(
            gates * torch.log(gates + eps)
            + (1 - gates) * torch.log(1 - gates + eps)
        )
        entropy_loss = entropy.mean()

        return self.gate_balance_weight * (balance_loss + entropy_loss)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        ecg_embedding: torch.Tensor,
        tabular_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            ecg_embedding: ECG embedding ``(batch_size, ecg_dim)``
            tabular_embedding: Tabular embedding ``(batch_size, tabular_dim)``

        Returns:
            fused_embedding: ``(batch_size, output_dim)``
            fusion_info: Dictionary containing:
                * **gate_summary** ``(B, 2)`` -- per-sample ``[ecg_weight,
                  tabular_weight]`` averaged across dimensions (backward-
                  compatible with the original scalar-gate interface).
                * **element_wise_gates** ``(B, output_dim)`` -- full
                  element-wise gate values for fine-grained analysis.
                * **auxiliary_loss** -- scalar regularisation loss to add to
                  the main task loss.
        """
        # 1. Project each modality to the shared space (tanh-bounded)
        h_ecg = self.ecg_transform(ecg_embedding)           # (B, output_dim)
        h_tabular = self.tabular_transform(tabular_embedding)  # (B, output_dim)

        # 2. Cross-modal interaction (optional)
        if self.use_cross_interaction:
            h_ecg, h_tabular = self._cross_interact(h_ecg, h_tabular)

        # 3. Compute element-wise gates from *raw* embeddings
        gate_input = torch.cat([ecg_embedding, tabular_embedding], dim=-1)
        gates = self.gate_network(gate_input)  # (B, output_dim) in [0, 1]

        # 4. Gated fusion: gate → 1 ⇒ ECG, gate → 0 ⇒ tabular
        fused = gates * h_ecg + (1 - gates) * h_tabular  # (B, output_dim)

        # 5. Output normalisation + dropout
        fused = self.output_norm(fused)
        fused = self.output_dropout(fused)

        # ── Interpretability & auxiliary outputs ─────────────────────────
        ecg_weight = gates.mean(dim=-1, keepdim=True)       # (B, 1)
        tabular_weight = 1 - ecg_weight                     # (B, 1)
        gate_summary = torch.cat([ecg_weight, tabular_weight], dim=-1)  # (B, 2)

        auxiliary_loss = self.compute_auxiliary_loss(gates)

        fusion_info = {
            'gate_summary': gate_summary,          # (B, 2) backward-compat
            'element_wise_gates': gates,           # (B, output_dim) full detail
            'auxiliary_loss': auxiliary_loss,       # scalar
        }

        return fused, fusion_info


class MultiLabelPredictionHead(nn.Module):
    """
    Multi-label prediction head with temperature scaling for calibration.
    
    Args:
        input_dim: Input embedding dimension
        num_labels: Number of labels to predict
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        use_temperature: Whether to use temperature scaling
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_labels: int = 12,
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
        use_temperature: bool = True
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.use_temperature = use_temperature
        
        # Prediction network
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, num_labels))
        
        self.network = nn.Sequential(*layers)
        
        # Temperature parameter for calibration (learned during training)
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input embedding (batch_size, input_dim)
            return_logits: Whether to return logits in addition to probabilities
            
        Returns:
            Dictionary containing:
                - probs: Predicted probabilities (batch_size, num_labels)
                - logits: Raw logits (batch_size, num_labels) if return_logits=True
                - calibrated_probs: Temperature-scaled probabilities
        """
        # Get logits
        logits = self.network(x)  # (B, num_labels)
        
        # Apply temperature scaling for calibration
        if self.use_temperature:
            calibrated_logits = logits / self.temperature
        else:
            calibrated_logits = logits
        
        # Convert to probabilities
        probs = torch.sigmoid(logits)
        calibrated_probs = torch.sigmoid(calibrated_logits)
        
        output = {
            'probs': probs,
            'calibrated_probs': calibrated_probs,
        }
        
        if return_logits:
            output['logits'] = logits
            
        return output


class SHDMultimodalModel(nn.Module):
    """
    Complete Structural Heart Disease multimodal prediction model.
    
    Combines ECG encoder, tabular encoder, fusion module, and prediction head.
    
    Args:
        ecg_config: Configuration for ECG encoder
        tabular_config: Configuration for tabular encoder
        fusion_config: Configuration for fusion module
        prediction_config: Configuration for prediction head
    """
    
    def __init__(
        self,
        ecg_config: Optional[dict] = None,
        tabular_config: Optional[dict] = None,
        fusion_config: Optional[dict] = None,
        prediction_config: Optional[dict] = None
    ):
        super().__init__()
        
        # Default configurations
        ecg_config = ecg_config or {}
        tabular_config = tabular_config or {}
        fusion_config = fusion_config or {}
        prediction_config = prediction_config or {}
        
        # Initialize encoders
        self.ecg_encoder = ECGTransformerEncoder(**ecg_config)
        self.tabular_encoder = TabularEncoder(**tabular_config)
        
        # Get embedding dimensions
        ecg_dim = self.ecg_encoder.embed_dim
        tabular_dim = self.tabular_encoder.embed_dim
        
        # Initialize fusion
        fusion_config.setdefault('ecg_dim', ecg_dim)
        fusion_config.setdefault('tabular_dim', tabular_dim)
        self.fusion = GatedFusion(**fusion_config)
        
        # Initialize prediction head
        prediction_config.setdefault('input_dim', fusion_config.get('output_dim', 256))
        self.prediction_head = MultiLabelPredictionHead(**prediction_config)
        
        # Enable MC Dropout for uncertainty estimation
        self.enable_mc_dropout = False
        
    def forward(
        self,
        waveform: torch.Tensor,
        tabular: torch.Tensor,
        tabular_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            waveform: ECG waveform (batch_size, 1, 2500, 12)
            tabular: Tabular features (batch_size, 7)
            tabular_mask: Missingness mask (batch_size, 7)
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing predictions and optional embeddings.
            Always includes ``auxiliary_loss`` from the gated-fusion module
            (a scalar that should be added to the main task loss).
        """
        # Encode ECG
        ecg_embedding, ecg_sequence = self.ecg_encoder(waveform)
        
        # Encode tabular data
        tabular_embedding, tabular_info = self.tabular_encoder(tabular, tabular_mask)
        
        # Fuse modalities (returns dict with gate_summary, element_wise_gates, auxiliary_loss)
        fused_embedding, fusion_info = self.fusion(ecg_embedding, tabular_embedding)
        
        # Predict
        predictions = self.prediction_head(fused_embedding, return_logits=True)
        
        # Prepare output
        output = {
            'probs': predictions['probs'],
            'calibrated_probs': predictions['calibrated_probs'],
            'logits': predictions['logits'],
            # (B, 2) summary: [ecg_weight, tabular_weight] -- backward-compatible
            'fusion_gates': fusion_info['gate_summary'],
            # Scalar auxiliary loss from gated-fusion regularisation
            'auxiliary_loss': fusion_info['auxiliary_loss'],
        }
        
        if return_embeddings:
            output.update({
                'ecg_embedding': ecg_embedding,
                'tabular_embedding': tabular_embedding,
                'fused_embedding': fused_embedding,
                'ecg_sequence': ecg_sequence,
                'tabular_info': tabular_info,
                # Full element-wise gates for fine-grained analysis
                'element_wise_gates': fusion_info['element_wise_gates'],
            })
        
        return output
    
    def predict_with_uncertainty(
        self,
        waveform: torch.Tensor,
        tabular: torch.Tensor,
        tabular_mask: torch.Tensor,
        num_samples: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            waveform: ECG waveform (batch_size, 1, 2500, 12)
            tabular: Tabular features (batch_size, 7)
            tabular_mask: Missingness mask (batch_size, 7)
            num_samples: Number of MC dropout samples
            
        Returns:
            Dictionary with mean predictions and uncertainty estimates
        """
        self.train()  # Enable dropout
        
        # Collect predictions from multiple forward passes
        all_probs = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.forward(waveform, tabular, tabular_mask)
                all_probs.append(output['calibrated_probs'])
        
        # Stack predictions
        all_probs = torch.stack(all_probs, dim=0)  # (num_samples, batch_size, num_labels)
        
        # Compute mean and uncertainty
        mean_probs = all_probs.mean(dim=0)
        uncertainty = all_probs.std(dim=0)  # Predictive uncertainty
        
        self.eval()  # Back to eval mode
        
        return {
            'mean_probs': mean_probs,
            'uncertainty': uncertainty,
            'all_samples': all_probs
        }
