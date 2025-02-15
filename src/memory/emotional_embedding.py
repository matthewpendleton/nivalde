"""Emotional state processing with memory integration.

This module implements the emotional state processing that combines
BERT-contextualized current input with historical context from memory.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class EmotionalProcessor(nn.Module):
    """Processes emotional state by combining current and historical context."""
    
    def __init__(self, dim: int = 768):
        """Initialize the emotional processor.
        
        Args:
            dim: Dimension of the embeddings
        """
        super().__init__()
        
        # Process combined context
        self.context_processor = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            nn.LayerNorm(2*dim),
            nn.ReLU(),
            nn.Linear(2*dim, dim)
        )
        
        # State integration
        self.state_integrator = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh()
        )
        
        self.previous_state = None
        
    def forward(self,
                bert_context: torch.Tensor,
                memory_context: torch.Tensor,
                prev_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process current input with historical context.
        
        Args:
            bert_context: BERT-contextualized current input
            memory_context: Historical context from memory
            prev_state: Optional previous emotional state
            
        Returns:
            Tuple of (new_state, processed_context)
        """
        # Process current and historical context
        combined_context = torch.cat([bert_context, memory_context], dim=-1)
        processed_context = self.context_processor(combined_context)
        
        # Integrate with previous state
        if prev_state is None:
            prev_state = torch.zeros_like(bert_context)
            
        state_input = torch.cat([prev_state, processed_context], dim=-1)
        new_state = self.state_integrator(state_input)
        
        self.previous_state = new_state
        return new_state, processed_context
