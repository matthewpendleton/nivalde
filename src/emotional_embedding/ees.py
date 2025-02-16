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
        hidden_dim: int = 256,
        latent_dim: int = 32
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoders for different input types
        self.bert_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.memory_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Emotional embedding layer
        self.emotional_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.previous_state = None
        self.previous_latent = None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent space."""
        if x.size(-1) != self.latent_dim:
            return self.bert_encoder(x)
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from learned emotional space."""
        return self.decoder(z)
    
    def forward(
        self,
        current_state: torch.Tensor,
        bert_context: torch.Tensor,
        memory_context: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the emotional embedding space."""
        # Ensure inputs have batch dimension
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        if bert_context.dim() == 1:
            bert_context = bert_context.unsqueeze(0)
        if memory_context.dim() == 1:
            memory_context = memory_context.unsqueeze(0)
        
        # Encode inputs into latent space
        current_encoded = self.encode(current_state)
        bert_encoded = self.encode(bert_context)
        memory_encoded = self.encode(memory_context)
        
        # Combine encoded representations
        combined = torch.cat([bert_encoded, memory_encoded], dim=-1)
        
        # Generate emotional embedding
        emotional_embedding = self.emotional_layer(combined)
        
        # Store both raw and latent states
        self.previous_state = current_state.detach()
        self.previous_latent = emotional_embedding.detach()
        
        return emotional_embedding
    
    def emotional_transition_loss(self, prev_state, curr_state):
        """Calculate how smooth the emotional transition is between states."""
        if prev_state is None or curr_state is None:
            return torch.tensor(0.0, device=curr_state.device)
        
        # Ensure both states are in latent space
        prev_encoded = self.encode(prev_state)
        curr_encoded = self.encode(curr_state)
        
        # Ensure batch dimension
        if prev_encoded.dim() == 1:
            prev_encoded = prev_encoded.unsqueeze(0)
        if curr_encoded.dim() == 1:
            curr_encoded = curr_encoded.unsqueeze(0)
        
        # Calculate cosine similarity between encoded states
        similarity = F.cosine_similarity(prev_encoded, curr_encoded)
        
        # We want smooth transitions (high similarity) but not static states
        transition_loss = torch.mean((1 - similarity) ** 2)
        
        return transition_loss
    
    def contextual_alignment_loss(self, curr_state, context_states):
        """Calculate how well the current state aligns with surrounding context."""
        if not context_states:
            return torch.tensor(0.0, device=curr_state.device)
            
        # Ensure current state has batch dimension
        if curr_state.dim() == 1:
            curr_state = curr_state.unsqueeze(0)
            
        # Process each context state
        context_losses = []
        for context in context_states:
            if context is None:
                continue
                
            if context.dim() == 1:
                context = context.unsqueeze(0)
            
            # Project context into same space as current state if needed
            context_encoded = context
            if context.size(-1) != self.latent_dim:
                context_encoded = self.bert_encoder(context)
                
            # Project current state if needed
            curr_encoded = curr_state
            if curr_state.size(-1) != self.latent_dim:
                curr_encoded = self.bert_encoder(curr_state)
            
            # Calculate alignment using cosine similarity
            similarity = F.cosine_similarity(curr_encoded, context_encoded)
            context_losses.append(similarity)
        
        if not context_losses:
            return torch.tensor(0.0, device=curr_state.device)
            
        # Average alignment across all context states
        avg_alignment = torch.stack(context_losses).mean()
        
        # We want moderate alignment with context (not too high or too low)
        # This loss encourages states to be contextually appropriate but not identical
        alignment_loss = torch.mean((0.7 - avg_alignment) ** 2)
        
        return alignment_loss
    
    def compute_loss(
        self,
        current_state: torch.Tensor,
        bert_context: torch.Tensor,
        memory_context: torch.Tensor
    ) -> torch.Tensor:
        """Compute the total loss for training."""
        # Get the emotional embedding
        latent = self.forward(current_state, bert_context, memory_context)
        
        # Ensure dimensions match for reconstruction
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        
        # Reconstruction loss
        reconstructed = self.decoder(latent)
        recon_loss = F.mse_loss(reconstructed, current_state)
        
        # Emotional transition loss using latent states
        transition_loss = self.emotional_transition_loss(
            self.previous_latent,  # Use previous latent state
            latent  # Current latent state
        )
        
        # Context alignment loss
        context_loss = self.contextual_alignment_loss(
            latent,  # Current latent state
            [bert_context, memory_context]  # Raw context states
        )
        
        # Combine losses with weights
        total_loss = recon_loss + 0.3 * transition_loss + 0.3 * context_loss
        
        return total_loss
