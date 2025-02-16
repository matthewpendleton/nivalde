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
        momentum (float): Momentum coefficient for state updates
        mixing_ratio (float): Ratio for mixing new context with previous state
    """
    
    def __init__(self, dim: int = 768, momentum: float = 0.8, mixing_ratio: float = 0.2):
        """Initialize the EES.
        
        Args:
            dim: Dimension of the emotional embedding space
            momentum: Momentum coefficient for state updates (0 to 1)
            mixing_ratio: Ratio for mixing new context with previous state (0 to 1)
        """
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.mixing_ratio = mixing_ratio
        
        # Process current and historical context
        self.context_processor = nn.Sequential(
            nn.Linear(2*dim, 2*dim),  # Combine current and historical
            nn.LayerNorm(2*dim),
            nn.ReLU(),
            nn.Linear(2*dim, dim),
            nn.LayerNorm(dim),  # Normalize context
            nn.Tanh()  # Bound context values
        )
        
        # State integration with hysteresis
        self.state_integrator = nn.Sequential(
            nn.Linear(2*dim, dim),  # Combine previous state and context
            nn.LayerNorm(dim),
            nn.Tanh(),  # Bound state values
            nn.Linear(dim, dim),  # Additional transformation for stability
            nn.LayerNorm(dim),
            nn.Sigmoid()  # Final activation for smooth convergence
        )
        
        # Previous state and momentum
        self.register_buffer('prev_state', None)
        self.register_buffer('velocity', None)
        
        # Decay factor for momentum
        self.decay_factor = 0.95
    
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
        # Initialize or use provided previous state
        if prev_state is not None:
            self.prev_state = prev_state
        elif self.prev_state is None:
            self.prev_state = torch.zeros_like(current_state)
            self.velocity = torch.zeros_like(current_state)
            
        # Process current and historical context
        combined_context = torch.cat([bert_context, memory_context], dim=-1)
        processed_context = self.context_processor(combined_context)
        
        # Mix processed context with previous state
        mixed_context = (
            self.mixing_ratio * processed_context + 
            (1 - self.mixing_ratio) * self.prev_state
        )
        
        # Compute state update with momentum
        state_input = torch.cat([self.prev_state, mixed_context], dim=-1)
        state_update = self.state_integrator(state_input)
        
        # Apply momentum with decay
        if self.velocity is not None:
            # Decay momentum over time
            effective_momentum = self.momentum * self.decay_factor
            
            self.velocity = (
                effective_momentum * self.velocity + 
                (1 - effective_momentum) * (state_update - self.prev_state)
            )
            new_state = self.prev_state + self.velocity
        else:
            new_state = state_update
            self.velocity = state_update - self.prev_state
        
        # Update previous state
        self.prev_state = new_state
        
        # Normalize output state
        new_state = new_state / (new_state.norm() + 1e-6)
        
        return new_state
