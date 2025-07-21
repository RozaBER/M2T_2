"""
Residual Vector Quantization (RVQ) Module for Brain-to-Text Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class VectorQuantizer(nn.Module):
    """Single vector quantizer with codebook"""
    def __init__(self, 
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 epsilon: float = 1e-5,
                 use_ema: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        
        # Initialize codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)
        
        if use_ema:
            # EMA parameters
            self.decay = decay
            self.epsilon = epsilon
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_w', torch.zeros(embedding_dim, num_embeddings))
            self._ema_initted = False
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [batch, seq_len, embedding_dim] or [batch, embedding_dim]
            
        Returns:
            quantized: Quantized outputs
            indices: Codebook indices
            commitment_loss: Commitment loss for training
        """
        # Handle both 2D and 3D inputs
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Get nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(input_shape)
        
        # Compute losses
        if self.training:
            if self.use_ema:
                # EMA update
                self._ema_update(flat_input, encodings)
                commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
            else:
                # Standard VQ loss
                e_latent_loss = F.mse_loss(quantized.detach(), inputs)
                q_latent_loss = F.mse_loss(quantized, inputs.detach())
                commitment_loss = self.commitment_cost * e_latent_loss + q_latent_loss
        else:
            commitment_loss = torch.tensor(0.0, device=inputs.device)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Reshape indices
        indices = encoding_indices.view(input_shape[:-1])
        
        return quantized, indices, commitment_loss
    
    def _ema_update(self, flat_input: torch.Tensor, encodings: torch.Tensor):
        """Update codebook with exponential moving average"""
        if not self._ema_initted:
            self.ema_w.data.copy_(flat_input.t() @ encodings)
            self.ema_cluster_size.data.copy_(encodings.sum(0))
            self._ema_initted = True
        else:
            self.ema_cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            
            # Laplace smoothing
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size.data.add_(self.epsilon)
            self.ema_cluster_size.data.div_(n + self.num_embeddings * self.epsilon).mul_(n)
            
            dw = torch.matmul(flat_input.t(), encodings)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            self.embeddings.weight.data.copy_(
                (self.ema_w / self.ema_cluster_size.unsqueeze(0)).t()
            )


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization module that progressively quantizes vectors
    using multiple codebooks to capture fine details
    """
    def __init__(self,
                 embedding_dim: int = 512,
                 num_quantizers: int = 8,
                 codebook_size: int = 256,
                 commitment_cost: float = 0.2,
                 decay: float = 0.99,
                 use_ema: bool = True,
                 quantizer_dropout: float = 0.0):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout
        
        # Create quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=codebook_size,
                commitment_cost=commitment_cost,
                decay=decay,
                use_ema=use_ema
            ) for _ in range(num_quantizers)
        ])
        
        # Codebook usage tracking
        self.register_buffer('codebook_usage', torch.zeros(num_quantizers, codebook_size))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass of RVQ
        
        Args:
            x: Input tensor [batch, seq_len, embedding_dim]
            
        Returns:
            quantized: Final quantized output
            indices: Codebook indices [batch, seq_len, num_quantizers]
            aux_loss: Dictionary containing losses
        """
        batch_size, seq_len, _ = x.shape
        
        quantized = torch.zeros_like(x)
        residual = x
        indices_list = []
        commitment_losses = []
        
        # Determine active quantizers (for dropout during training)
        if self.training and self.quantizer_dropout > 0:
            active_quantizers = torch.rand(self.num_quantizers) > self.quantizer_dropout
            active_quantizers[0] = True  # Always use at least the first quantizer
        else:
            active_quantizers = torch.ones(self.num_quantizers, dtype=torch.bool)
        
        # Progressive quantization
        for i, quantizer in enumerate(self.quantizers):
            if not active_quantizers[i]:
                # Skip this quantizer (dropout)
                indices_list.append(torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device))
                continue
                
            # Quantize residual
            quantized_residual, indices, commitment_loss = quantizer(residual)
            
            # Update quantized vector
            quantized = quantized + quantized_residual
            
            # Compute new residual
            residual = residual - quantized_residual.detach()
            
            # Store results
            indices_list.append(indices)
            commitment_losses.append(commitment_loss)
            
            # Update usage statistics
            if self.training:
                self.codebook_usage[i].scatter_add_(
                    0, indices.flatten(), torch.ones_like(indices.flatten(), dtype=torch.float)
                )
        
        # Stack indices
        indices = torch.stack(indices_list, dim=-1)  # [batch, seq_len, num_quantizers]
        
        # Compute total commitment loss
        total_commitment_loss = sum(commitment_losses) / len(commitment_losses) if commitment_losses else 0.0
        
        # Compute diversity loss to encourage codebook usage
        diversity_loss = self._compute_diversity_loss()
        
        aux_loss = {
            'commitment_loss': total_commitment_loss,
            'diversity_loss': diversity_loss,
            'total_vq_loss': total_commitment_loss + 0.1 * diversity_loss
        }
        
        return quantized, indices, aux_loss
    
    def _compute_diversity_loss(self) -> torch.Tensor:
        """Compute diversity loss to encourage uniform codebook usage"""
        # Normalize usage counts
        usage = self.codebook_usage / (self.codebook_usage.sum(dim=1, keepdim=True) + 1e-5)
        
        # Compute entropy
        entropy = -torch.sum(usage * torch.log(usage + 1e-10), dim=1)
        
        # Maximum entropy
        max_entropy = torch.log(torch.tensor(self.codebook_size, dtype=torch.float))
        
        # Diversity loss is negative normalized entropy
        diversity_loss = 1.0 - entropy.mean() / max_entropy
        
        return diversity_loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to discrete codes only"""
        _, indices, _ = self.forward(x)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete codes back to continuous representation"""
        batch_size, seq_len, num_quantizers = indices.shape
        
        quantized = torch.zeros(
            batch_size, seq_len, self.embedding_dim, 
            device=indices.device, dtype=torch.float
        )
        
        for i, quantizer in enumerate(self.quantizers):
            if i < num_quantizers:
                # Get embeddings for indices
                quantized_i = quantizer.embeddings(indices[:, :, i])
                quantized = quantized + quantized_i
        
        return quantized
    
    def reset_usage_stats(self):
        """Reset codebook usage statistics"""
        self.codebook_usage.zero_()
    
    def get_codebook_usage(self) -> dict:
        """Get codebook usage statistics"""
        usage_stats = {}
        total_usage = self.codebook_usage.sum()
        
        for i in range(self.num_quantizers):
            usage = self.codebook_usage[i]
            usage_stats[f'quantizer_{i}'] = {
                'usage_counts': usage.cpu().numpy(),
                'usage_rate': (usage > 0).float().mean().item(),
                'entropy': -torch.sum(
                    (usage / usage.sum()) * torch.log((usage / usage.sum()) + 1e-10)
                ).item() if usage.sum() > 0 else 0.0
            }
        
        return usage_stats


class RVQWithReconstruction(nn.Module):
    """RVQ with optional reconstruction decoder for self-supervised pretraining"""
    def __init__(self,
                 embedding_dim: int = 512,
                 num_quantizers: int = 8,
                 codebook_size: int = 256,
                 n_channels: int = 208,
                 use_reconstruction: bool = True):
        super().__init__()
        
        self.rvq = ResidualVectorQuantizer(
            embedding_dim=embedding_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size
        )
        
        self.use_reconstruction = use_reconstruction
        if use_reconstruction:
            # Reconstruction decoder (transpose convolution)
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(embedding_dim, 256, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose1d(256, 128, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, n_channels, 4, 2, 1),
            )
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> dict:
        """
        Forward pass with optional reconstruction
        
        Args:
            x: Original input signal [batch, channels, time]
            encoder_output: Encoder output [batch, seq_len, embedding_dim]
            
        Returns:
            Dictionary with quantized output, indices, and losses
        """
        quantized, indices, vq_losses = self.rvq(encoder_output)
        
        result = {
            'quantized': quantized,
            'indices': indices,
            'vq_losses': vq_losses
        }
        
        if self.use_reconstruction and self.training:
            # Reconstruct signal
            quantized_t = quantized.transpose(1, 2)  # [batch, embedding_dim, seq_len]
            reconstructed = self.decoder(quantized_t)
            
            # Compute reconstruction loss
            # Adjust dimensions if needed
            min_len = min(reconstructed.size(2), x.size(2))
            reconstruction_loss = F.mse_loss(
                reconstructed[:, :, :min_len], 
                x[:, :, :min_len]
            )
            
            result['reconstructed'] = reconstructed
            result['reconstruction_loss'] = reconstruction_loss
        
        return result
