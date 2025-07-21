"""
Full Brain-to-Text Model combining EEG Encoder, RVQ, and LLM Decoder
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
from .eeg_encoder import EEGEncoder
from .rvq_module_optimized import OptimizedResidualVectorQuantizer, RVQWithReconstruction
from .llm_decoder import BrainToTextLLM, BrainToTextConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class BrainToTextModel(nn.Module):
    """
    End-to-end Brain-to-Text model that maps EEG/MEG signals to text
    
    Architecture:
    EEG/MEG signals -> EEG Encoder -> RVQ Quantizer -> LLM Decoder -> Text
    """
    
    def __init__(self,
                 # EEG Encoder parameters
                 n_channels: int = 208,
                 sampling_rate: int = 1000,
                 segment_length: float = 0.1,
                 eeg_d_model: int = 512,
                 eeg_n_heads: int = 8,
                 eeg_n_layers: int = 6,
                 eeg_d_ff: int = 2048,
                 eeg_dropout: float = 0.1,
                 use_moe: bool = False,
                 num_experts: int = 8,
                 # RVQ parameters
                 num_quantizers: int = 8,
                 codebook_size: int = 256,
                 commitment_cost: float = 0.2,
                 use_ema: bool = True,
                 use_reconstruction: bool = False,
                 # LLM parameters
                 vocab_size: int = 32000,
                 llm_hidden_size: int = 1024,
                 llm_intermediate_size: int = 4096,
                 llm_n_layers: int = 12,
                 llm_n_heads: int = 16,
                 max_position_embeddings: int = 2048,
                 # Training parameters
                 freeze_encoder: bool = False,
                 freeze_llm: bool = False):
        super().__init__()
        
        # Store configuration
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self._freeze_encoder = freeze_encoder
        self._freeze_llm = freeze_llm
        
        # Initialize EEG Encoder
        self.eeg_encoder = EEGEncoder(
            n_channels=n_channels,
            sampling_rate=sampling_rate,
            segment_length=segment_length,
            d_model=eeg_d_model,
            n_heads=eeg_n_heads,
            n_layers=eeg_n_layers,
            d_ff=eeg_d_ff,
            dropout=eeg_dropout,
            use_moe=use_moe,
            num_experts=num_experts
        )
        
        # Initialize RVQ module
        if use_reconstruction:
            self.rvq = RVQWithReconstruction(
                embedding_dim=eeg_d_model,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                n_channels=n_channels,
                use_reconstruction=use_reconstruction
            )
        else:
            self.rvq = OptimizedResidualVectorQuantizer(
                embedding_dim=eeg_d_model,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                commitment_cost=commitment_cost,
                use_ema=use_ema
            )
        
        # Initialize LLM Decoder
        llm_config = BrainToTextConfig(
            vocab_size=vocab_size,
            hidden_size=llm_hidden_size,
            intermediate_size=llm_intermediate_size,
            num_hidden_layers=llm_n_layers,
            num_attention_heads=llm_n_heads,
            max_position_embeddings=max_position_embeddings,
            eeg_token_start_id=vocab_size,
            num_eeg_tokens=num_quantizers * codebook_size,
            eeg_hidden_size=eeg_d_model
        )
        self.llm_decoder = BrainToTextLLM(llm_config)
        
        # Apply freezing if requested
        if freeze_encoder:
            for param in self.eeg_encoder.parameters():
                param.requires_grad = False
        
        if freeze_llm:
            for param in self.llm_decoder.parameters():
                param.requires_grad = False
    
    def encode_eeg(self, eeg_signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Encode EEG signals to discrete tokens
        
        Args:
            eeg_signals: [batch, channels, time]
            
        Returns:
            quantized: Quantized embeddings [batch, seq_len, embedding_dim]
            indices: Token indices [batch, seq_len, num_quantizers]
            aux_losses: Dictionary of auxiliary losses
        """
        # Encode EEG signals
        encoder_output = self.eeg_encoder(eeg_signals)
        
        # Quantize with RVQ
        if isinstance(self.rvq, RVQWithReconstruction):
            rvq_output = self.rvq(eeg_signals, encoder_output)
            quantized = rvq_output['quantized']
            indices = rvq_output['indices']
            aux_losses = rvq_output['vq_losses']
            if 'reconstruction_loss' in rvq_output:
                aux_losses['reconstruction_loss'] = rvq_output['reconstruction_loss']
        else:
            quantized, indices, aux_losses = self.rvq(encoder_output)
        
        return quantized, indices, aux_losses
    
    def forward(self,
                eeg_signals: Optional[torch.Tensor] = None,
                eeg_indices: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                return_dict: Optional[bool] = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete model
        
        Args:
            eeg_signals: Raw EEG/MEG signals [batch, channels, time]
            eeg_indices: Pre-computed EEG token indices [batch, seq_len, num_quantizers]
            input_ids: Text token IDs for teacher forcing [batch, text_len]
            attention_mask: Attention mask for text tokens
            labels: Target text token IDs for loss computation
            use_cache: Whether to use KV cache
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing:
                - loss: Total loss (if labels provided)
                - lm_loss: Language modeling loss
                - vq_loss: Vector quantization loss
                - logits: Output logits
                - eeg_indices: Quantized EEG indices
                - quantized_eeg: Quantized EEG embeddings
        """
        total_loss = 0.0
        aux_losses = {}
        
        # Process EEG signals if provided
        if eeg_signals is not None:
            quantized_eeg, eeg_indices, vq_losses = self.encode_eeg(eeg_signals)
            aux_losses.update(vq_losses)
        else:
            quantized_eeg = None
        
        # Forward pass through LLM
        llm_output = self.llm_decoder(
            input_ids=input_ids,
            eeg_indices=eeg_indices,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            return_dict=True
        )
        
        # Combine losses with optimized weights
        lm_loss = llm_output.loss if llm_output.loss is not None else torch.tensor(0.0)
        vq_loss = aux_losses.get('total_vq_loss', torch.tensor(0.0))
        recon_loss = aux_losses.get('reconstruction_loss', torch.tensor(0.0))
        commitment_loss = aux_losses.get('commitment_loss', torch.tensor(0.0))
        diversity_loss = aux_losses.get('diversity_loss', torch.tensor(0.0))
        perplexity_loss = aux_losses.get('perplexity_loss', torch.tensor(0.0))
        
        # Optimized loss weighting
        total_loss = (
            1.0 * lm_loss +
            0.5 * recon_loss +
            0.2 * commitment_loss +
            0.1 * diversity_loss +
            0.05 * perplexity_loss
        )
        
        if return_dict:
            return {
                'loss': total_loss,
                'lm_loss': lm_loss,
                'vq_loss': vq_loss,
                'reconstruction_loss': recon_loss,
                'commitment_loss': commitment_loss,
                'diversity_loss': diversity_loss,
                'perplexity_loss': perplexity_loss,
                'logits': llm_output.logits,
                'eeg_indices': eeg_indices,
                'quantized_eeg': quantized_eeg,
                'past_key_values': llm_output.past_key_values if use_cache else None,
                'aux_losses': aux_losses
            }
        else:
            return (total_loss, llm_output.logits, eeg_indices, quantized_eeg)
    
    def generate(self,
                 eeg_signals: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 do_sample: bool = True,
                 pad_token_id: int = 0,
                 eos_token_id: int = 2) -> torch.Tensor:
        """
        Generate text from EEG signals
        
        Args:
            eeg_signals: Raw EEG/MEG signals [batch, channels, time]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token IDs [batch, seq_len]
        """
        # Encode EEG to discrete tokens
        with torch.no_grad():
            _, eeg_indices, _ = self.encode_eeg(eeg_signals)
        
        # Generate text using LLM
        generated_ids = self.llm_decoder.generate_from_eeg(
            eeg_indices=eeg_indices,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id
        )
        
        return generated_ids
    
    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save({
            'eeg_encoder_state_dict': self.eeg_encoder.state_dict(),
            'rvq_state_dict': self.rvq.state_dict(),
            'llm_decoder_state_dict': self.llm_decoder.state_dict(),
            'config': {
                'n_channels': self.n_channels,
                'sampling_rate': self.sampling_rate,
                'segment_length': self.segment_length,
                'num_quantizers': self.num_quantizers,
                'codebook_size': self.codebook_size,
                'eeg_d_model': self.eeg_encoder.d_model,
                'llm_config': self.llm_decoder.config.to_dict()
            }
        }, os.path.join(save_directory, 'model.pt'))
        
        # Save LLM config separately for HuggingFace compatibility
        self.llm_decoder.config.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from saved weights"""
        import os
        
        # Load checkpoint
        checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')
        config = checkpoint['config']
        
        # Extract all necessary config parameters
        # Count encoder layers from state dict
        encoder_state = checkpoint['eeg_encoder_state_dict']
        encoder_layers = max([int(k.split('.')[2]) for k in encoder_state.keys() 
                            if k.startswith('transformer.layers.')], default=5) + 1
        
        # Extract LLM config parameters
        llm_config = config.get('llm_config', {})
        vocab_size = llm_config.get('vocab_size', 32000)
        llm_hidden_size = llm_config.get('hidden_size', 1024)
        llm_intermediate_size = llm_config.get('intermediate_size', 4096)
        llm_n_layers = llm_config.get('num_hidden_layers', 12)
        llm_n_heads = llm_config.get('num_attention_heads', 16)
        
        # Create model instance with full config
        model = cls(
            n_channels=config['n_channels'],
            sampling_rate=config['sampling_rate'],
            segment_length=config['segment_length'],
            num_quantizers=config['num_quantizers'],
            codebook_size=config['codebook_size'],
            eeg_d_model=config['eeg_d_model'],
            eeg_n_layers=encoder_layers,
            vocab_size=vocab_size,
            llm_hidden_size=llm_hidden_size,
            llm_intermediate_size=llm_intermediate_size,
            llm_n_layers=llm_n_layers,
            llm_n_heads=llm_n_heads,
            **kwargs
        )
        
        # Load state dicts
        model.eeg_encoder.load_state_dict(checkpoint['eeg_encoder_state_dict'])
        model.rvq.load_state_dict(checkpoint['rvq_state_dict'])
        model.llm_decoder.load_state_dict(checkpoint['llm_decoder_state_dict'])
        
        return model
    
    def get_num_params(self, only_trainable: bool = False) -> int:
        """Get number of parameters"""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def freeze_encoder(self):
        """Freeze EEG encoder parameters"""
        for param in self.eeg_encoder.parameters():
            param.requires_grad = False
        self._freeze_encoder = True
    
    def unfreeze_encoder(self):
        """Unfreeze EEG encoder parameters"""
        for param in self.eeg_encoder.parameters():
            param.requires_grad = True
        self._freeze_encoder = False
    
    def freeze_llm(self):
        """Freeze LLM decoder parameters"""
        for param in self.llm_decoder.parameters():
            param.requires_grad = False
        self._freeze_llm = True
    
    def unfreeze_llm(self):
        """Unfreeze LLM decoder parameters"""
        for param in self.llm_decoder.parameters():
            param.requires_grad = True
        self._freeze_llm = False
