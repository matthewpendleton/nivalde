"""Emotional Embedding Space implementation.

This module implements the emotional state representation that combines
current contextualized input with historical context from memory.
"""

import torch
import torch.nn as nn

class EmotionalEmbeddingSpace(nn.Module):
    """Emotional embedding space that maintains state hysteresis through
    integration of current and historical context.
    
    Attributes:
        dim (int): Dimension of the emotional embedding space
        context_processor (nn.Module): Processes current and historical context
        state_integrator (nn.Module): Integrates state updates with hysteresis
    """
    
    def __init__(self, dim: int = 768):
        """Initialize the EES.
        
        Args:
            dim: Dimension of the emotional embedding space
        """
        super().__init__()
        self.dim = dim
        
        # Process current and historical context
        self.context_processor = nn.Sequential(
            nn.Linear(2*dim, 2*dim),  # Combine current and historical
            nn.LayerNorm(2*dim),
            nn.ReLU(),
            nn.Linear(2*dim, dim)
        )
        
        # State integration with hysteresis
        self.state_integrator = nn.Sequential(
            nn.Linear(2*dim, dim),  # Combine previous state and context
            nn.LayerNorm(dim),
            nn.Tanh()
        )
    
    def forward(self, 
                current_state: torch.Tensor,
                bert_context: torch.Tensor,
                memory_context: torch.Tensor,
                prev_state: torch.Tensor = None) -> torch.Tensor:
        """Evolve the emotional state based on current and historical context.
        
        Args:
            current_state: Current emotional state
            bert_context: BERT-contextualized current input
            memory_context: Historical context from Transformer² memory
            prev_state: Previous emotional state (optional)
            
        Returns:
            Updated emotional state
        """
        # Process current and historical context
        combined_context = torch.cat([bert_context, memory_context], dim=-1)
        processed_context = self.context_processor(combined_context)
        
        # Integrate with previous state (if available)
        if prev_state is None:
            prev_state = torch.zeros_like(current_state)
            
        state_input = torch.cat([prev_state, processed_context], dim=-1)
        new_state = self.state_integrator(state_input)
        
        return new_state
