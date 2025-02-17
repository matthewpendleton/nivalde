"""Emotional Embedding Space (EES) implementation.

The EES is a natural manifold that emerges from the interaction between language
and memory. Each memory carries two temporal components:
1. When it was disclosed (t_disclosure)
2. When it occurred (t_reference)

Both temporal aspects influence how memories shape the emotional landscape.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass
import time

@dataclass
class TemporalMemory:
    """Memory with dual temporal components."""
    state: torch.Tensor          # The emotional content
    t_disclosure: float         # When they told us about it
    t_reference: Optional[float] # When it happened (if known)
    
class EmotionalEmbeddingSpace(nn.Module):
    """A lens into the natural emotional topology of language and memory.
    
    Each memory has two temporal components that affect its influence:
    1. t_disclosure: Recency of disclosure affects immediate impact
    2. t_reference: Historical depth affects root strength
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        disclosure_decay: float = 0.1,  # How fast recent disclosures fade
        reference_decay: float = 0.05,  # How fast historical roots fade
        disclosure_weight: float = 0.6,  # Impact of recent disclosures
        reference_weight: float = 0.4    # Impact of historical roots
    ):
        super().__init__()
        self.input_dim = input_dim
        self.disclosure_decay = disclosure_decay
        self.reference_decay = reference_decay
        self.disclosure_weight = disclosure_weight
        self.reference_weight = reference_weight
        
        # Store memories with their temporal information
        self.memories: list[TemporalMemory] = []
        
    def add_memory(
        self,
        state: torch.Tensor,
        t_reference: Optional[float] = None
    ) -> None:
        """Add a new memory with current disclosure time and optional reference time."""
        t_now = time.time()
        memory = TemporalMemory(
            state=state,
            t_disclosure=t_now,
            t_reference=t_reference
        )
        self.memories.append(memory)
        
    def compute_temporal_weights(self, t_now: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute disclosure and reference weights for all memories."""
        if not self.memories:
            return torch.tensor([]), torch.tensor([])
            
        # Disclosure time weights (recent disclosures matter more)
        t_disclosures = torch.tensor([m.t_disclosure for m in self.memories])
        disclosure_weights = torch.exp(
            -self.disclosure_decay * (t_now - t_disclosures)
        )
        
        # Reference time weights (deeper history can have stronger roots)
        t_references = torch.tensor([
            m.t_reference if m.t_reference is not None else m.t_disclosure 
            for m in self.memories
        ])
        reference_weights = torch.exp(
            -self.reference_decay * (t_now - t_references)
        )
        
        return disclosure_weights, reference_weights
        
    def forward(
        self,
        x: torch.Tensor,
        t_reference: Optional[float] = None
    ) -> torch.Tensor:
        """Observe the emotional state emerging from current input and temporal memory.
        
        Args:
            x: BERT embedding of current utterance (batch_size, input_dim)
            t_reference: Optional reference time for this utterance
            
        Returns:
            Current emotional state
        """
        # Add current state to memories
        self.add_memory(x, t_reference)
        
        # No previous memories yet
        if len(self.memories) == 1:
            return x
            
        t_now = time.time()
        disclosure_weights, reference_weights = self.compute_temporal_weights(t_now)
        
        # Normalize weights
        disclosure_weights = disclosure_weights / disclosure_weights.sum()
        reference_weights = reference_weights / reference_weights.sum()
        
        # Compute memory influence from both temporal aspects
        memory_states = torch.stack([m.state for m in self.memories[:-1]])  # Exclude current
        
        memory_influence = (
            self.disclosure_weight * (disclosure_weights[:-1, None] * memory_states).sum(0) +
            self.reference_weight * (reference_weights[:-1, None] * memory_states).sum(0)
        )
        
        # Blend current state with temporally-weighted memory
        emotional_state = x + memory_influence
        
        return emotional_state
    
    def get_memory_timeline(self) -> list[Tuple[float, float]]:
        """Get list of (disclosure_time, reference_time) pairs for visualization."""
        return [(m.t_disclosure, m.t_reference or m.t_disclosure) for m in self.memories]
