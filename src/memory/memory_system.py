"""Memory system implementation using standard Transformer² architecture.

This module implements hierarchical memory storage and retrieval based on
novelty/surprise signals, providing historical context to the EES.

Each memory includes both its disclosure time (when it was told) and
reference time (when it occurred), allowing the system to understand both
the recency of information and its historical depth.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, NamedTuple
import torch.nn.functional as F
from dataclasses import dataclass
import time
from .temporal_parser import TemporalParser, TemporalReference

@dataclass
class TemporalMemory:
    """Memory with dual temporal components and content."""
    state: torch.Tensor          # Emotional/semantic content
    t_disclosure: float          # When they told us
    t_reference: float          # When it occurred
    t_uncertainty: float        # Uncertainty in reference time (std dev in days)
    embedding: torch.Tensor      # Full temporal-enriched embedding
    confidence: float           # Overall confidence in memory

class TemporalUncertainty:
    """Helper class for handling vague temporal references."""
    
    # Common temporal expressions and their approximate uncertainties (in days)
    UNCERTAINTY_MAPPINGS = {
        "yesterday": 0.2,
        "last_week": 2.0,
        "last_month": 7.0,
        "few_months_ago": 30.0,
        "last_year": 60.0,
        "years_ago": 180.0,
        "childhood": 365.0,
        "long_ago": 730.0
    }
    
    @staticmethod
    def estimate_uncertainty(temporal_expr: str) -> float:
        """Estimate temporal uncertainty from natural language expression."""
        expr = temporal_expr.lower().replace(" ", "_")
        return TemporalUncertainty.UNCERTAINTY_MAPPINGS.get(expr, 90.0)  # Default to 3 months
    
    @staticmethod
    def gaussian_window(t: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """Create Gaussian temporal window."""
        return torch.exp(-0.5 * ((t - mean) / std) ** 2)

class Transformer2Memory(nn.Module):
    """Transformer²-based memory system with temporal awareness.
    
    Augments memory embeddings with temporal encoding that captures both:
    - When memories were disclosed (t_disclosure)
    - When they occurred (t_reference)
    
    This allows the attention mechanism to naturally learn temporal relationships
    and weighting patterns based on both time dimensions.
    """
    
    def __init__(
        self,
        dim: int = 384,          # Main embedding dimension
        temporal_dim: int = 64,   # Temporal encoding dimension
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_memories: int = 1000  # Maximum memories to store
    ):
        super().__init__()
        
        self.dim = dim
        self.temporal_dim = temporal_dim
        self.total_dim = dim + temporal_dim
        self.num_heads = num_heads
        self.max_memories = max_memories
        
        # Temporal encoding network
        self.temporal_encoder = nn.Sequential(
            nn.Linear(152, temporal_dim // 2),
            nn.LayerNorm(temporal_dim // 2),
            nn.ReLU(),
            nn.Linear(temporal_dim // 2, temporal_dim),
            nn.LayerNorm(temporal_dim)
        )
        
        # Initialize transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.total_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Memory storage
        self.memories: List[TemporalMemory] = []
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.total_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.temporal_parser = TemporalParser()
        
    def encode_temporal(
        self,
        t_disclosure: torch.Tensor,
        t_reference: torch.Tensor,
        t_uncertainty: torch.Tensor,
        t_now: Optional[float] = None
    ) -> torch.Tensor:
        """Encode temporal information with uncertainty."""
        if t_now is None:
            t_now = time.time()
            
        # Convert to relative times (in days)
        t_disclosure_rel = (t_now - t_disclosure) / (24 * 3600)  # Convert to days
        t_reference_rel = (t_now - t_reference) / (24 * 3600)
        
        # Create temporal uncertainty windows
        time_grid = torch.linspace(0, 365*5, 50)  # 5 year grid
        disclosure_window = TemporalUncertainty.gaussian_window(
            time_grid, 
            t_disclosure_rel, 
            torch.ones_like(t_disclosure_rel)  # Fixed 1-day uncertainty for disclosure
        )
        reference_window = TemporalUncertainty.gaussian_window(
            time_grid, 
            t_reference_rel, 
            t_uncertainty
        )
        
        # Combine windows with temporal features
        temporal_features = torch.cat([
            disclosure_window,
            reference_window,
            t_uncertainty.unsqueeze(-1)  # Include raw uncertainty
        ], dim=-1)
        
        return self.temporal_encoder(temporal_features)

    def create_memory(
        self,
        state: torch.Tensor,
        temporal_expr: Optional[str] = None,
        t_reference: Optional[float] = None,
        confidence: float = 1.0,
        context_name: Optional[str] = None
    ) -> TemporalMemory:
        """Create a new memory with sophisticated temporal parsing.
        
        Args:
            state: Emotional state tensor
            temporal_expr: Natural language time expression
            t_reference: Explicit timestamp (if known)
            confidence: Base confidence in memory
            context_name: Name to store as temporal context anchor
        """
        t_now = time.time()
        
        if temporal_expr is not None:
            # Parse temporal expression
            temporal_ref = self.temporal_parser.parse_expression(
                temporal_expr,
                base_confidence=confidence
            )
            
            # Store as context anchor if name provided
            if context_name:
                self.temporal_parser.add_context(context_name, temporal_ref)
            
            t_ref = temporal_ref.mean_time
            t_uncertainty = temporal_ref.uncertainty
            confidence = temporal_ref.confidence
            
        else:
            t_ref = t_reference if t_reference is not None else t_now
            t_uncertainty = 1.0  # Default 1 day uncertainty
        
        # Create temporal encoding with uncertainty
        temporal_features = self.encode_temporal(
            torch.tensor([t_now]),
            torch.tensor([t_ref]),
            torch.tensor([t_uncertainty])
        )
        
        # Combine with state embedding
        full_embedding = torch.cat([state, temporal_features[0]], dim=-1)
        
        memory = TemporalMemory(
            state=state,
            t_disclosure=t_now,
            t_reference=t_ref,
            t_uncertainty=t_uncertainty,
            embedding=full_embedding,
            confidence=confidence
        )
        
        if len(self.memories) >= self.max_memories:
            self.memories.pop(0)
        self.memories.append(memory)
        
        return memory
        
    def get_memory_batch(self) -> torch.Tensor:
        """Get batch of all memory embeddings for transformer processing."""
        if not self.memories:
            return torch.empty(0, self.total_dim)
            
        return torch.stack([m.embedding for m in self.memories])
        
    def forward(
        self,
        query_state: torch.Tensor,
        t_reference: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """Process query through temporal memory system.
        
        Args:
            query_state: Current emotional state [batch_size, dim]
            t_reference: Optional reference time for query
            
        Returns:
            Tuple of:
            - Updated state with memory context
            - Attention weights for each memory
        """
        # Create query embedding with temporal encoding
        query_memory = self.create_memory(query_state, t_reference=t_reference)
        memory_batch = self.get_memory_batch()
        
        if memory_batch.size(0) == 0:
            return query_state, []
            
        # Add query to memory batch for self-attention
        full_batch = torch.cat([
            memory_batch,
            query_memory.embedding.unsqueeze(0)
        ])
        
        # Process through transformer
        transformed = self.transformer(full_batch)
        
        # Temporal attention between query and memories
        attn_output, attn_weights = self.temporal_attention(
            transformed[-1:],  # Query
            transformed[:-1],  # Keys (memories)
            transformed[:-1]   # Values (memories)
        )
        
        # Combine with original state
        output_state = query_state + attn_output[0, :self.dim]
        
        return output_state, attn_weights[0].tolist()


class TransformerMemorySystem(nn.Module):
    """High-level interface for the Transformer² memory system."""
    
    def __init__(self, embedding_dim: int = 768):
        """Initialize the memory system.
        
        Args:
            embedding_dim: Dimension of input embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention = nn.Linear(embedding_dim, embedding_dim)
        self.gate_network = nn.Linear(2 * embedding_dim, embedding_dim)
        self.novelty_threshold = 0.5
        
        # Initialize memory buffer
        self.register_buffer('memory_buffer', 
            torch.zeros(32, embedding_dim)  # Start with 32 memory slots
        )
        self.register_buffer('memory_mask',
            torch.zeros(32, dtype=torch.bool)  # Track which slots are used
        )

    def forward(
        self,
        current_state: torch.Tensor,
        memory_context: torch.Tensor
    ) -> torch.Tensor:
        """Process current state and update memory context.
        
        Args:
            current_state: Current input state [batch_size, input_dim]
            memory_context: Previous memory context [batch_size, input_dim]
        
        Returns:
            Updated memory context [batch_size, input_dim]
        """
        # Ensure inputs have batch dimension
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        if memory_context.dim() == 1:
            memory_context = memory_context.unsqueeze(0)
        
        # Validate input dimensions
        batch_size, input_dim = current_state.size()
        if input_dim != self.embedding_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.embedding_dim}, got {input_dim}")
        
        # If no memories are used yet, initialize with current state
        if not self.memory_mask.any():
            new_memory_buffer = self.memory_buffer.clone()
            new_memory_mask = self.memory_mask.clone()
            
            # Initialize first memory slot with current state
            new_memory_buffer[0] = current_state[0].detach()
            new_memory_mask[0] = True
            
            self.memory_buffer = new_memory_buffer
            self.memory_mask = new_memory_mask
        
        # Get active memories
        active_memories = self.memory_buffer[self.memory_mask]
        if len(active_memories) == 0:
            # If somehow we still have no memories, just return the input
            return current_state
        
        # Compute attention over active memories
        query = self.attention(current_state)  # [batch_size, embedding_dim]
        attention_scores = torch.matmul(query, active_memories.t())  # [batch_size, num_active]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weight and sum memories
        weighted_memories = torch.matmul(attention_weights, active_memories)  # [batch_size, embedding_dim]
        
        # Combine with previous context using gating
        gate = torch.sigmoid(self.gate_network(torch.cat([weighted_memories, memory_context], dim=-1)))
        new_context = gate * weighted_memories + (1 - gate) * memory_context
        
        # Update memories if current state is novel enough
        novelty = self.compute_novelty(current_state, weighted_memories)
        update_mask = novelty > self.novelty_threshold
        
        if update_mask.any():
            # Find empty slots or overwrite oldest memories
            empty_slots = (~self.memory_mask).nonzero(as_tuple=True)[0]
            num_updates = update_mask.sum().item()
            
            if len(empty_slots) > 0:
                # Use empty slots first
                slots_to_use = empty_slots[:num_updates]
                new_memory_buffer = self.memory_buffer.clone()
                new_memory_mask = self.memory_mask.clone()
                
                for i, slot in enumerate(slots_to_use):
                    if i < update_mask.sum():
                        new_memory_buffer[slot] = current_state[update_mask][i].detach()
                        new_memory_mask[slot] = True
                
                self.memory_buffer = new_memory_buffer
                self.memory_mask = new_memory_mask
            else:
                # Overwrite oldest memories (just use first available slots)
                slots_to_use = torch.arange(num_updates, device=self.memory_buffer.device)
                new_memory_buffer = self.memory_buffer.clone()
                
                for i, slot in enumerate(slots_to_use):
                    if i < update_mask.sum():
                        new_memory_buffer[slot] = current_state[update_mask][i].detach()
                
                self.memory_buffer = new_memory_buffer
        
        return new_context

    def compute_novelty(self, current_state: torch.Tensor, weighted_memories: torch.Tensor) -> torch.Tensor:
        """Compute novelty score between current state and weighted memories.
        
        Args:
            current_state: Current input state [batch_size, input_dim]
            weighted_memories: Weighted sum of memories [batch_size, input_dim]
        
        Returns:
            Novelty scores [batch_size]
        """
        return (current_state - weighted_memories).norm(dim=-1)
