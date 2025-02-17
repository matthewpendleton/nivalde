"""Emotional Embedding Space (EES) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class EmotionalEmbeddingSpace(nn.Module):
    """Learns emotional embeddings through unsupervised dimensional reduction.
    
    Instead of using predefined emotional dimensions, this implementation learns
    a latent emotional space through autoencoder-style dimensional reduction.
    The model discovers emotional patterns and relationships purely from the data.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        latent_dim: int = 128,  # Increased for more expressive embeddings
        num_heads: int = 8,     # Multi-head attention
        dropout_rate: float = 0.2,
        l2_weight: float = 1e-4,
        recon_weight: float = 1.0,
        transition_weight: float = 0.3,
        context_weight: float = 0.3,
        contrastive_weight: float = 0.1  # New contrastive learning weight
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.recon_weight = recon_weight
        self.transition_weight = transition_weight
        self.context_weight = context_weight
        self.contrastive_weight = contrastive_weight
        self.l2_weight = l2_weight
        
        # Hierarchical encoder with attention
        self.encoder_local = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.encoder_global = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Emotion-specific projection
        self.emotion_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Hierarchical decoder
        self.decoder_global = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.decoder_local = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.previous_state = None
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Hierarchical encoding with attention."""
        # Local features
        local_features = self.encoder_local(x)
        
        # Self-attention for context
        if local_features.dim() == 2:
            local_features = local_features.unsqueeze(1)
        
        attended_features, _ = self.attention(
            local_features, local_features, local_features
        )
        
        # Global features
        global_features = self.encoder_global(attended_features.squeeze(1))
        
        # Emotion-specific projection
        emotional_features = self.emotion_projector(global_features)
        
        return emotional_features
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Hierarchical decoding."""
        # Global features
        global_features = self.decoder_global(z)
        
        # Local reconstruction
        reconstructed = self.decoder_local(global_features)
        
        return reconstructed
    
    def compute_contrastive_loss(self, current_embedding, next_embedding):
        """Compute contrastive loss for better emotional transitions."""
        # Normalize embeddings
        current_norm = F.normalize(current_embedding, dim=1)
        next_norm = F.normalize(next_embedding, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(current_norm, next_norm.t())
        
        # Temperature scaling
        temperature = 0.07
        similarity = similarity / temperature
        
        # Labels are the diagonal (positive pairs)
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        
        # Compute contrastive loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def forward(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        memory_system: nn.Module
    ) -> torch.Tensor:
        """Forward pass through the emotional embedding space.
        
        Args:
            sequences: Batch of sequences [batch_size, max_seq_len, input_dim]
            lengths: Length of each sequence [batch_size]
            memory_system: Memory system module
            
        Returns:
            Loss value for the batch
        """
        batch_size = sequences.size(0)
        total_loss = torch.tensor(0.0, device=sequences.device)
        
        # Process each sequence in the batch
        for i in range(batch_size):
            seq_len = lengths[i]
            sequence = sequences[i, :seq_len]  # [seq_len, input_dim]
            
            # Initialize memory context
            memory_context = torch.zeros(self.input_dim, device=sequence.device)
            
            # Process each utterance in the sequence
            for j in range(seq_len):
                current_state = sequence[j]  # Current utterance
                
                # Get BERT context (previous utterance if available)
                bert_context = sequence[j-1] if j > 0 else torch.zeros_like(current_state)
                
                # Get memory context from memory system
                memory_context = memory_system(current_state.unsqueeze(0), memory_context.unsqueeze(0))
                memory_context = memory_context.squeeze(0)
                
                # Compute loss for this utterance
                loss = self.compute_loss(
                    current_state,
                    bert_context,
                    memory_context
                )
                
                total_loss += loss
        
        # Average loss over batch
        return total_loss / batch_size
    
    def emotional_transition_loss(self, prev_state, curr_state):
        """Calculate how smooth the emotional transition is between states."""
        # Encode both states
        prev_latent = self.encode(prev_state)
        curr_latent = self.encode(curr_state)
        
        # Calculate transition smoothness using MSE
        transition_loss = F.mse_loss(curr_latent, prev_latent, reduction='mean')
        
        return transition_loss
    
    def contextual_alignment_loss(self, curr_state: torch.Tensor, context_state: torch.Tensor):
        """Calculate how well the current state aligns with surrounding context."""
        # Encode both states
        curr_latent = self.encode(curr_state)
        context_latent = self.encode(context_state)
        
        # Calculate alignment using cosine similarity
        alignment = 1 - F.cosine_similarity(curr_latent, context_latent, dim=1).mean()
        
        return alignment
    
    def compute_loss(
        self,
        current_state: torch.Tensor,
        bert_context: torch.Tensor,
        memory_context: torch.Tensor
    ) -> torch.Tensor:
        """Compute the total loss for training."""
        logger = logging.getLogger(__name__)
        
        try:
            # Ensure dimensions match for reconstruction
            if current_state.dim() == 1:
                current_state = current_state.unsqueeze(0)
            if bert_context.dim() == 1:
                bert_context = bert_context.unsqueeze(0)
            if memory_context.dim() == 1:
                memory_context = memory_context.unsqueeze(0)
            
            # Log shapes for debugging
            logger.debug(f"Current state shape: {current_state.shape}")
            logger.debug(f"BERT context shape: {bert_context.shape}")
            logger.debug(f"Memory context shape: {memory_context.shape}")
            
            # Check for NaN or Inf in inputs
            if torch.isnan(current_state).any() or torch.isinf(current_state).any():
                logger.error("NaN or Inf found in current_state")
                return torch.tensor(100.0, device=current_state.device)
            
            # Ensure all tensors are on the same device
            device = current_state.device
            bert_context = bert_context.to(device)
            memory_context = memory_context.to(device)
            
            # Get the emotional embedding
            latent = self.encode(current_state)
            if torch.isnan(latent).any() or torch.isinf(latent).any():
                logger.error("NaN or Inf found in latent encoding")
                return torch.tensor(100.0, device=device)
            
            # Reconstruction loss
            reconstructed = self.decode(latent)
            if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                logger.error("NaN or Inf found in reconstruction")
                return torch.tensor(100.0, device=device)
            
            recon_loss = F.mse_loss(reconstructed, current_state, reduction='mean')
            recon_loss = torch.clamp(recon_loss, min=0.0, max=10.0)
            
            # Emotional transition loss
            transition_loss = self.emotional_transition_loss(bert_context, current_state)
            transition_loss = torch.clamp(transition_loss, min=0.0, max=10.0)
            
            # Context alignment loss
            context_loss = self.contextual_alignment_loss(current_state, memory_context)
            context_loss = torch.clamp(context_loss, min=0.0, max=10.0)
            
            # Contrastive loss
            contrastive_loss = self.compute_contrastive_loss(latent, self.previous_latent)
            contrastive_loss = torch.clamp(contrastive_loss, min=0.0, max=10.0)
            
            # L2 regularization
            l2_reg = torch.tensor(0., device=device)
            for param in self.parameters():
                if not torch.isnan(param).any() and not torch.isinf(param).any():
                    l2_reg += torch.norm(param)
            l2_reg = torch.clamp(l2_reg, min=0.0, max=10.0)
            
            # Total loss with regularization
            total_loss = (
                self.recon_weight * recon_loss +
                self.transition_weight * transition_loss +
                self.context_weight * context_loss +
                self.contrastive_weight * contrastive_loss +
                self.l2_weight * l2_reg
            )
            
            # Final safety clamp
            total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
            
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                logger.error(f"Invalid total loss: {total_loss.item()}")
                logger.error(f"Components: recon={recon_loss.item()}, trans={transition_loss.item()}, context={context_loss.item()}, contrastive={contrastive_loss.item()}, l2={l2_reg.item()}")
                return torch.tensor(100.0, device=device)
            
            self.previous_latent = latent
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {str(e)}")
            return torch.tensor(100.0, device=device)
