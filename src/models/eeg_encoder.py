"""
EEG Encoder with CNN + Transformer architecture for Brain-to-Text Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ConvFeatureExtractor(nn.Module):
    """1D Convolutional feature extractor for EEG/MEG signals"""
    def __init__(self, 
                 n_channels: int,
                 conv_channels: list = [64, 128, 256],
                 kernel_sizes: list = [7, 5, 3],
                 strides: list = [2, 2, 2],
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        in_channels = n_channels
        
        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(conv_channels, kernel_sizes, strides)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*layers)
        self.output_dim = conv_channels[-1]
        
    def forward(self, x):
        # x: [batch, channels, time]
        return self.conv_layers(x)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer for scaling EEG encoder"""
    def __init__(self, 
                 dim: int,
                 num_experts: int = 8,
                 expert_dim: int = 2048,
                 top_k: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network
        self.gate = nn.Linear(dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, dim)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        # Compute gating scores
        gate_scores = self.gate(x_flat)
        gate_scores = F.softmax(gate_scores, dim=-1)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Apply experts
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_score = top_k_scores[:, i].unsqueeze(-1)
            
            # Gather samples for each expert
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_score[mask] * expert_output
        
        return output.view(batch_size, seq_len, dim)


class EEGEncoder(nn.Module):
    """
    EEG Encoder that captures spatiotemporal patterns from EEG/MEG signals
    using CNN + Transformer architecture with optional MoE layers
    """
    def __init__(self,
                 n_channels: int = 208,  # MEG channels
                 sampling_rate: int = 1000,
                 segment_length: float = 0.1,  # 100ms segments
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 use_moe: bool = False,
                 num_experts: int = 8):
        super().__init__()
        
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.d_model = d_model
        
        # CNN feature extractor
        self.conv_extractor = ConvFeatureExtractor(
            n_channels=n_channels,
            conv_channels=[128, 256, 512],
            kernel_sizes=[7, 5, 3],
            strides=[2, 2, 2]
        )
        
        # Project CNN output to d_model
        self.input_projection = nn.Linear(self.conv_extractor.output_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Optional MoE layers
        self.use_moe = use_moe
        if use_moe:
            self.moe_layers = nn.ModuleList([
                MixtureOfExperts(d_model, num_experts, d_ff, top_k=2)
                for _ in range(n_layers // 2)
            ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of EEG encoder
        
        Args:
            x: Input EEG/MEG signals [batch, channels, time]
            mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Encoded features [batch, seq_len, d_model]
        """
        # Extract CNN features
        conv_out = self.conv_extractor(x)  # [batch, conv_channels, reduced_time]
        
        # Reshape for transformer: [batch, seq_len, features]
        conv_out = conv_out.transpose(1, 2)
        
        # Project to d_model
        features = self.input_projection(conv_out)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Transformer encoding
        if self.use_moe:
            # Interleave transformer and MoE layers
            for i in range(len(self.transformer.layers)):
                features = self.transformer.layers[i](features, src_key_padding_mask=mask)
                if i % 2 == 1 and i // 2 < len(self.moe_layers):
                    features = self.moe_layers[i // 2](features)
        else:
            features = self.transformer(features, src_key_padding_mask=mask)
        
        # Final layer norm
        features = self.layer_norm(features)
        
        return features
    
    def get_segment_embeddings(self, x: torch.Tensor, 
                             segment_samples: Optional[int] = None) -> torch.Tensor:
        """
        Get embeddings for fixed-length segments of EEG
        
        Args:
            x: Input EEG/MEG signals [batch, channels, time]
            segment_samples: Number of samples per segment
            
        Returns:
            Segment embeddings [batch, n_segments, d_model]
        """
        if segment_samples is None:
            segment_samples = int(self.segment_length * self.sampling_rate)
        
        batch_size, n_channels, total_samples = x.shape
        n_segments = total_samples // segment_samples
        
        # Reshape into segments
        x_segments = x[:, :, :n_segments * segment_samples].reshape(
            batch_size, n_channels, n_segments, segment_samples
        )
        
        # Process each segment
        embeddings = []
        for i in range(n_segments):
            segment = x_segments[:, :, i, :]
            segment_embedding = self.forward(segment)
            # Average pool over time dimension
            segment_embedding = segment_embedding.mean(dim=1)
            embeddings.append(segment_embedding)
        
        return torch.stack(embeddings, dim=1)
