"""
Optimized Residual Vector Quantization (RVQ) Module for Brain-to-Text Model
Fixes codebook collapse issues with k-means++ initialization and diversity mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np
from sklearn.cluster import KMeans


class OptimizedVectorQuantizer(nn.Module):
    """Enhanced vector quantizer with k-means initialization and temperature annealing"""
    def __init__(self, 
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 epsilon: float = 1e-5,
                 use_ema: bool = True,
                 temperature: float = 1.0,
                 gradient_scale: float = 10.0,
                 dead_code_threshold: int = 100):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.temperature = temperature
        self.gradient_scale = gradient_scale
        self.dead_code_threshold = dead_code_threshold
        
        # Initialize codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)
        
        # Track codebook usage
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        self.register_buffer('steps_since_last_use', torch.zeros(num_embeddings))
        self.register_buffer('initialized', torch.tensor(False))
        
        if use_ema:
            # EMA parameters
            self.decay = decay
            self.epsilon = epsilon
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_w', torch.zeros(embedding_dim, num_embeddings))
            self._ema_initted = False
        
    def kmeans_init(self, data: torch.Tensor):
        """Initialize codebook using k-means++ on first batch"""
        if self.initialized:
            return
            
        with torch.no_grad():
            # Flatten data for clustering
            flat_data = data.view(-1, self.embedding_dim).cpu().numpy()
            
            # Subsample if too large
            if len(flat_data) > 10000:
                indices = np.random.choice(len(flat_data), 10000, replace=False)
                flat_data = flat_data[indices]
            
            # Check if we have enough samples for k-means
            if len(flat_data) >= self.num_embeddings:
                # Run k-means++
                kmeans = KMeans(n_clusters=self.num_embeddings, init='k-means++', n_init=1)
                kmeans.fit(flat_data)
                
                # Update embeddings
                self.embeddings.weight.data.copy_(torch.from_numpy(kmeans.cluster_centers_))
            else:
                # Not enough samples, use random initialization from the data
                print(f"Warning: Not enough samples ({len(flat_data)}) for k-means with {self.num_embeddings} clusters. Using random initialization.")
                if len(flat_data) > 0:
                    # Use available samples and duplicate/interpolate to fill codebook
                    indices = np.random.choice(len(flat_data), self.num_embeddings, replace=True)
                    centers = flat_data[indices]
                    # Add small random noise to duplicated centers
                    noise = np.random.normal(0, 0.01, centers.shape)
                    centers = centers + noise
                    self.embeddings.weight.data.copy_(torch.from_numpy(centers))
                else:
                    # Fallback to random normal initialization
                    self.embeddings.weight.data.normal_(0, 0.02)
            
            self.initialized.fill_(True)
    
    def forward(self, inputs: torch.Tensor, temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with temperature-based selection
        
        Args:
            inputs: [batch, seq_len, embedding_dim] or [batch, embedding_dim]
            temperature: Override default temperature for inference
            
        Returns:
            quantized: Quantized outputs
            indices: Codebook indices
            commitment_loss: Commitment loss for training
        """
        # Initialize with k-means on first forward pass
        if not self.initialized and self.training:
            self.kmeans_init(inputs)
        
        # Handle both 2D and 3D inputs
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Apply temperature scaling
        temp = temperature if temperature is not None else self.temperature
        if temp != 1.0 and self.training:
            distances = distances / temp
        
        # Get nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize with gradient scaling
        quantized = torch.matmul(encodings, self.embeddings.weight)
        if self.training and self.gradient_scale != 1.0:
            quantized = quantized * self.gradient_scale + quantized.detach() * (1 - self.gradient_scale)
        
        quantized = quantized.view(input_shape)
        
        # Update usage statistics
        if self.training:
            self.code_usage.scatter_add_(0, encoding_indices, torch.ones_like(encoding_indices, dtype=torch.float))
            self.steps_since_last_use += 1
            self.steps_since_last_use.scatter_(0, encoding_indices, torch.zeros_like(encoding_indices, dtype=torch.float))
        
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
                
            # Random restart for dead codes
            self._revive_dead_codes(flat_input)
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
    
    def _revive_dead_codes(self, inputs: torch.Tensor):
        """Revive dead codes by reinitializing them with random inputs"""
        dead_codes = self.steps_since_last_use > self.dead_code_threshold
        n_dead = dead_codes.sum().item()
        
        if n_dead > 0:
            with torch.no_grad():
                # Sample random inputs
                n_samples = min(n_dead * 5, len(inputs))
                random_indices = torch.randperm(len(inputs))[:n_samples]
                random_samples = inputs[random_indices]
                
                # Reinitialize dead codes
                dead_indices = torch.where(dead_codes)[0]
                for i, idx in enumerate(dead_indices[:len(random_samples)]):
                    self.embeddings.weight.data[idx] = random_samples[i]
                    self.steps_since_last_use[idx] = 0


class OptimizedResidualVectorQuantizer(nn.Module):
    """
    Optimized RVQ module with enhanced diversity mechanisms
    """
    def __init__(self,
                 embedding_dim: int = 512,
                 num_quantizers: int = 8,
                 codebook_size: int = 8192,
                 commitment_cost: float = 0.2,
                 decay: float = 0.99,
                 use_ema: bool = True,
                 quantizer_dropout: float = 0.0,
                 initial_temperature: float = 2.0,
                 min_temperature: float = 0.5,
                 temperature_decay: float = 0.9999,
                 gradient_scale: float = 10.0,
                 diversity_weight: float = 0.1,
                 perplexity_weight: float = 0.05):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout
        self.diversity_weight = diversity_weight
        self.perplexity_weight = perplexity_weight
        
        # Temperature annealing
        self.register_buffer('temperature', torch.tensor(initial_temperature))
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        
        # Create quantizers with temperature
        self.quantizers = nn.ModuleList([
            OptimizedVectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=codebook_size,
                commitment_cost=commitment_cost,
                decay=decay,
                use_ema=use_ema,
                temperature=initial_temperature,
                gradient_scale=gradient_scale
            ) for _ in range(num_quantizers)
        ])
        
        # Codebook usage tracking
        self.register_buffer('codebook_usage', torch.zeros(num_quantizers, codebook_size))
        self.register_buffer('total_steps', torch.tensor(0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass of optimized RVQ
        
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
        perplexities = []
        
        # Update temperature
        if self.training:
            self.temperature.mul_(self.temperature_decay).clamp_(min=self.min_temperature)
            self.total_steps += 1
        
        # Determine active quantizers (for dropout during training)
        if self.training and self.quantizer_dropout > 0:
            active_quantizers = torch.rand(self.num_quantizers, device=x.device) > self.quantizer_dropout
            active_quantizers[0] = True  # Always use at least the first quantizer
        else:
            active_quantizers = torch.ones(self.num_quantizers, dtype=torch.bool, device=x.device)
        
        # Progressive quantization
        for i, quantizer in enumerate(self.quantizers):
            if not active_quantizers[i]:
                # Skip this quantizer (dropout)
                indices_list.append(torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device))
                continue
                
            # Quantize residual with current temperature
            quantized_residual, indices, commitment_loss = quantizer(residual, temperature=self.temperature.item())
            
            # Update quantized vector
            quantized = quantized + quantized_residual
            
            # Compute new residual
            residual = residual - quantized_residual.detach()
            
            # Store results
            indices_list.append(indices)
            commitment_losses.append(commitment_loss)
            
            # Calculate perplexity
            if self.training:
                # Update usage statistics
                self.codebook_usage[i].scatter_add_(
                    0, indices.flatten(), torch.ones_like(indices.flatten(), dtype=torch.float)
                )
                
                # Calculate perplexity for this quantizer
                usage = quantizer.code_usage / (quantizer.code_usage.sum() + 1e-10)
                perplexity = torch.exp(-torch.sum(usage * torch.log(usage + 1e-10)))
                perplexities.append(perplexity)
        
        # Stack indices
        indices = torch.stack(indices_list, dim=-1)  # [batch, seq_len, num_quantizers]
        
        # Compute losses
        total_commitment_loss = sum(commitment_losses) / len(commitment_losses) if commitment_losses else 0.0
        
        # Compute diversity loss
        diversity_loss = self._compute_diversity_loss()
        
        # Compute perplexity loss
        if perplexities:
            avg_perplexity = sum(perplexities) / len(perplexities)
            perplexity_loss = (self.codebook_size - avg_perplexity) / self.codebook_size
        else:
            perplexity_loss = torch.tensor(0.0, device=x.device)
        
        # Total VQ loss
        total_vq_loss = (
            total_commitment_loss + 
            self.diversity_weight * diversity_loss + 
            self.perplexity_weight * perplexity_loss
        )
        
        aux_loss = {
            'commitment_loss': total_commitment_loss,
            'diversity_loss': diversity_loss,
            'perplexity_loss': perplexity_loss,
            'total_vq_loss': total_vq_loss,
            'temperature': self.temperature.item(),
            'avg_perplexity': avg_perplexity.item() if perplexities else 0.0
        }
        
        return quantized, indices, aux_loss
    
    def _compute_diversity_loss(self) -> torch.Tensor:
        """Enhanced diversity loss to encourage uniform codebook usage"""
        diversity_losses = []
        
        for i in range(self.num_quantizers):
            # Get usage for this quantizer
            usage = self.codebook_usage[i]
            
            # Normalize
            if usage.sum() > 0:
                usage_prob = usage / usage.sum()
                
                # Compute entropy
                entropy = -torch.sum(usage_prob * torch.log(usage_prob + 1e-10))
                
                # Maximum entropy
                max_entropy = torch.log(torch.tensor(self.codebook_size, dtype=torch.float))
                
                # Diversity loss is negative normalized entropy
                diversity_loss = 1.0 - entropy / max_entropy
                diversity_losses.append(diversity_loss)
        
        if diversity_losses:
            return sum(diversity_losses) / len(diversity_losses)
        else:
            return torch.tensor(0.0, device=self.codebook_usage.device)
    
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
        for quantizer in self.quantizers:
            quantizer.code_usage.zero_()
            quantizer.steps_since_last_use.zero_()
    
    def get_codebook_usage(self) -> dict:
        """Get detailed codebook usage statistics"""
        usage_stats = {}
        total_usage = self.codebook_usage.sum()
        
        for i in range(self.num_quantizers):
            usage = self.codebook_usage[i]
            active_codes = (usage > 0).sum().item()
            usage_stats[f'quantizer_{i}'] = {
                'active_codes': active_codes,
                'usage_rate': active_codes / self.codebook_size,
                'usage_counts': usage.cpu().numpy(),
                'perplexity': torch.exp(-torch.sum(
                    (usage / (usage.sum() + 1e-10)) * torch.log((usage / (usage.sum() + 1e-10)) + 1e-10)
                )).item() if usage.sum() > 0 else 0.0,
                'entropy': -torch.sum(
                    (usage / (usage.sum() + 1e-10)) * torch.log((usage / (usage.sum() + 1e-10)) + 1e-10)
                ).item() if usage.sum() > 0 else 0.0
            }
        
        usage_stats['overall'] = {
            'temperature': self.temperature.item(),
            'total_steps': self.total_steps.item()
        }
        
        return usage_stats


class RVQWithReconstruction(nn.Module):
    """RVQ with optional reconstruction decoder for self-supervised pretraining"""
    def __init__(self,
                 embedding_dim: int = 512,
                 num_quantizers: int = 8,
                 codebook_size: int = 8192,
                 n_channels: int = 208,
                 use_reconstruction: bool = True):
        super().__init__()
        
        self.rvq = OptimizedResidualVectorQuantizer(
            embedding_dim=embedding_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size
        )
        
        self.use_reconstruction = use_reconstruction
        if use_reconstruction:
            # Enhanced reconstruction decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(embedding_dim, 512, 4, 2, 1),
                nn.GroupNorm(32, 512),
                nn.GELU(),
                nn.ConvTranspose1d(512, 256, 4, 2, 1),
                nn.GroupNorm(16, 256),
                nn.GELU(),
                nn.ConvTranspose1d(256, 128, 4, 2, 1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
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