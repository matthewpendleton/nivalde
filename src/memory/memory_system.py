"""Memory system implementation using standard Transformer² architecture.

This module implements hierarchical memory storage and retrieval based on
novelty/surprise signals, providing historical context to the EES.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple

class Transformer2Memory(nn.Module):
    """Standard Transformer² implementation for hierarchical memory storage.
    
    This implementation follows the original Transformer² paper, using
    hierarchical storage based on novelty/surprise signals.
    """
    
    def __init__(self, 
                 dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 dropout: float = 0.1):
        """Initialize the memory system.
        
        Args:
            dim: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Standard Transformer² encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=4*dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Memory storage
        self.memories = []
        self.surprise_scores = []
        
    def compute_surprise(self, 
                        new_memory: torch.Tensor,
                        prior_memories: Optional[torch.Tensor] = None) -> float:
        """Compute surprise/novelty score for new memory.
        
        The score is based on the negative log probability of the memory
        given prior memories.
        
        Args:
            new_memory: New memory embedding
            prior_memories: Optional tensor of prior memories
            
        Returns:
            Surprise score for the new memory
        """
        if prior_memories is None or len(prior_memories) == 0:
            return 1.0  # Maximum surprise for first memory
            
        # Compute attention scores with prior memories
        attn_scores = torch.matmul(
            new_memory, 
            prior_memories.transpose(-2, -1)
        ) / math.sqrt(new_memory.size(-1))
        
        # Negative log probability as surprise score
        prob = torch.softmax(attn_scores, dim=-1).max()
        return -torch.log(prob).item()
    
    def store_memory(self, memory: torch.Tensor):
        """Store new memory with its surprise score.
        
        Args:
            memory: New memory embedding to store
        """
        # Compute surprise score
        prior_memories = (torch.stack(self.memories) 
                         if self.memories else None)
        surprise = self.compute_surprise(memory, prior_memories)
        
        # Store memory and score
        self.memories.append(memory)
        self.surprise_scores.append(surprise)
        
        # Keep memories sorted by surprise score
        if len(self.memories) > 1:
            indices = torch.argsort(
                torch.tensor(self.surprise_scores),
                descending=True
            )
            self.memories = [self.memories[i] for i in indices]
            self.surprise_scores = [self.surprise_scores[i] for i in indices]
    
    def get_historical_context(self, 
                             current_input: torch.Tensor,
                             max_memories: int = 100) -> torch.Tensor:
        """Retrieve relevant historical context for current input.
        
        Args:
            current_input: Current input embedding
            max_memories: Maximum number of memories to consider
            
        Returns:
            Processed historical context
        """
        if not self.memories:
            return torch.zeros_like(current_input)
        
        # Get stored memories (limited by max_memories)
        memories = torch.stack(self.memories[:max_memories])
        
        # Add current input as first token
        sequence = torch.cat([
            current_input.unsqueeze(0),
            memories
        ], dim=0)
        
        # Process through transformer
        context = self.transformer(sequence)
        
        # Return processed context
        return context[0]  # First token contains contextualized representation
