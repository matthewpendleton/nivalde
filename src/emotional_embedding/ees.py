"""Emotional Embedding Space (EES) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionalEmbeddingSpace(nn.Module):
    """Learns emotional embeddings through unsupervised dimensional reduction.
    
    Instead of using predefined emotional dimensions, this implementation learns
    a latent emotional space through autoencoder-style dimensional reduction.
    The model discovers emotional patterns and relationships purely from the data.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Encoder learns compressed emotional representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder reconstructs original space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Memory integration layer
        self.memory_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        
        # State transition layer
        self.state_transition = nn.GRUCell(
            input_size=latent_dim,
            hidden_size=latent_dim
        )
        
        # Momentum factor for smooth transitions
        self.momentum = 0.8
        
        # Keep track of previous state
        self.previous_state = None
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into learned emotional space."""
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from learned emotional space."""
        return self.decoder(z)
    
    def forward(
        self,
        current_state: torch.Tensor,
        bert_context: torch.Tensor,
        memory_context: torch.Tensor
    ) -> torch.Tensor:
        """Process input through the emotional embedding space.
        
        Args:
            current_state: Current emotional state
            bert_context: BERT embedding of current input
            memory_context: Historical context from memory system
        
        Returns:
            New emotional state in learned latent space
        """
        # Encode current input
        current_encoded = self.encode(current_state)
        bert_encoded = self.encode(bert_context)
        memory_encoded = self.encode(memory_context)
        
        # Integrate memory context
        combined = torch.cat([bert_encoded, memory_encoded], dim=-1)
        memory_gate = self.memory_gate(combined)
        gated_memory = memory_gate * memory_encoded
        
        # Update state with memory-integrated input
        input_state = bert_encoded + gated_memory
        
        if self.previous_state is None:
            self.previous_state = torch.zeros_like(input_state)
            
        # Apply state transition with momentum
        new_state = self.state_transition(
            input_state,
            self.previous_state
        )
        
        new_state = (
            self.momentum * self.previous_state +
            (1 - self.momentum) * new_state
        )
        
        # Update previous state
        self.previous_state = new_state.detach()
        
        return new_state
        
    def compute_loss(
        self,
        current_state: torch.Tensor,
        bert_context: torch.Tensor,
        memory_context: torch.Tensor
    ) -> torch.Tensor:
        """Compute unsupervised learning loss.
        
        Combines reconstruction loss with temporal consistency
        to learn meaningful emotional dimensions.
        """
        # Get latent representation
        latent = self.forward(
            current_state,
            bert_context,
            memory_context
        )
        
        # Reconstruction loss
        reconstructed = self.decode(latent)
        recon_loss = F.mse_loss(reconstructed, current_state)
        
        # Temporal consistency loss if we have previous state
        temporal_loss = 0
        if self.previous_state is not None:
            temporal_loss = F.mse_loss(
                latent,
                self.previous_state
            )
        
        # Total loss
        total_loss = recon_loss + 0.1 * temporal_loss
        
        return total_loss
