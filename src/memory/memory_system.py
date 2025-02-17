"""Memory system implementation using standard Transformer² architecture.

This module implements hierarchical memory storage and retrieval based on
novelty/surprise signals, providing historical context to the EES.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple
import torch.nn.functional as F

class Transformer2Memory(nn.Module):
    """Standard Transformer² implementation for hierarchical memory storage.
    
    This implementation captures both emotional transitions and full contextual
    information, providing a comprehensive patient history to the EES.
    """
    
    def __init__(self, 
                 dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 dropout: float = 0.1,
                 temporal_decay: float = 0.1):  
        """Initialize the memory system.
        
        Args:
            dim: Model dimension (full context embedding size)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            temporal_decay: Temporal decay factor for surprise computation
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.temporal_decay = temporal_decay
        
        # Standard Transformer² encoder for full context
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
        
        # Separate attention for emotional analysis
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Surprise scoring network
        self.surprise_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Context integration network
        self.context_integrator = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Separate storage for emotional and full context
        self.memories = []  # Full context memories
        self.emotion_memories = []  # Emotional aspect memories
        self.surprise_scores = []
        
    def _init_weights(self):
        """Initialize network weights for better gradient flow."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                if param.dim() > 1:  # Only apply to weight matrices
                    nn.init.xavier_uniform_(param)
                else:  # For 1D parameters (like biases)
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def compute_temporal_weights(self, num_memories: int) -> torch.Tensor:
        """Compute temporal decay weights for memories.
        
        More recent memories get higher weights.
        
        Args:
            num_memories: Number of memories to compute weights for
            
        Returns:
            Tensor of temporal weights [num_memories]
        """
        if num_memories == 0:
            return torch.empty(0)
            
        times = torch.arange(num_memories, dtype=torch.float32)
        weights = torch.exp(-self.temporal_decay * (num_memories - 1 - times))
        return weights / weights.sum()  # Normalize
        
    def compute_surprise(self, 
                        new_memory: torch.Tensor,
                        prior_memories: Optional[torch.Tensor] = None) -> float:
        """Compute surprise score using attention and temporal weighting.
        
        Args:
            new_memory: New memory embedding [dim]
            prior_memories: Optional tensor of prior memories [num_memories, dim]
            
        Returns:
            Surprise score (0 to 1)
        """
        if prior_memories is None or len(prior_memories) == 0:
            return 0.5  # Moderate surprise for first memory
            
        # Ensure proper dimensions
        new_memory = new_memory.unsqueeze(0)  # [1, dim]
        if len(prior_memories.shape) == 1:
            prior_memories = prior_memories.unsqueeze(0)  # [1, dim]
            
        # Compute attention between new memory and prior memories
        with torch.no_grad():
            # Get attention weights
            attn_output, attn_weights = self.emotion_attention(
                new_memory,                    # query
                prior_memories,                # key
                prior_memories,                # value
                need_weights=True
            )
            
            # Get temporal weights [num_memories]
            temporal_weights = self.compute_temporal_weights(len(prior_memories))
            
            # Weight attention by temporal importance
            weighted_attn = attn_weights.squeeze(0) * temporal_weights
            
            # Get top-k most relevant memories (k=3)
            k = min(3, len(prior_memories))
            topk_weights, topk_indices = weighted_attn.topk(k)
            relevant_memories = prior_memories[topk_indices]
            
            # Compute rate of change between memories
            if len(relevant_memories) > 1:
                memory_diffs = torch.diff(relevant_memories, dim=0)
                avg_change_rate = torch.norm(memory_diffs, dim=1).mean()
            else:
                avg_change_rate = torch.tensor(0.0)
            
            # Current change magnitude
            current_diff = torch.norm(new_memory - relevant_memories[0])
            
            # Normalize the difference
            max_diff = torch.norm(torch.ones_like(new_memory) * 2)  # Maximum possible difference
            normalized_diff = current_diff / max_diff
            
            # Compare current change to historical rate
            if avg_change_rate == 0:
                surprise_score = normalized_diff
            else:
                # Exponential scaling for sudden changes
                change_ratio = (current_diff / (avg_change_rate + 1e-6)).clamp(0, 10)
                surprise_score = 1 - torch.exp(-change_ratio)
            
            # Weight by attention confidence and temporal recency
            attention_confidence = topk_weights.mean()
            temporal_factor = temporal_weights[-1]  # Most recent weight
            final_score = surprise_score * attention_confidence * temporal_factor
            
        return final_score.item()
        
    def store_memory(self, memory: torch.Tensor, emotion_embedding: Optional[torch.Tensor] = None):
        """Store new memory with both full context and emotional information.
        
        Args:
            memory: New memory embedding to store (full context)
            emotion_embedding: Optional emotional aspect of the memory
        """
        # Use the full memory as emotion embedding if none provided
        emotion_embedding = emotion_embedding if emotion_embedding is not None else memory
        
        # Compute surprise score using emotional embeddings
        prior_emotions = (torch.stack(self.emotion_memories) 
                         if self.emotion_memories else None)
        surprise = self.compute_surprise(emotion_embedding, prior_emotions)
        
        # Store both aspects of memory
        self.memories.append(memory)
        self.emotion_memories.append(emotion_embedding)
        self.surprise_scores.append(surprise)
        
        # Keep memories sorted by surprise score
        if len(self.memories) > 1:
            indices = torch.argsort(
                torch.tensor(self.surprise_scores),
                descending=True
            )
            self.memories = [self.memories[i] for i in indices]
            self.emotion_memories = [self.emotion_memories[i] for i in indices]
            self.surprise_scores = [self.surprise_scores[i] for i in indices]
    
    def get_historical_context(self, 
                             current_input: torch.Tensor,
                             max_memories: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve relevant historical context for current input.
        
        Args:
            current_input: Current input embedding
            max_memories: Maximum number of memories to consider
            
        Returns:
            Tuple of (processed_context, emotional_context)
        """
        if not self.memories:
            return torch.zeros_like(current_input), torch.zeros_like(current_input)
        
        # Get stored memories (limited by max_memories)
        memories = torch.stack(self.memories[:max_memories])
        emotion_memories = torch.stack(self.emotion_memories[:max_memories])
        
        # Process full context through transformer
        sequence = torch.cat([current_input.unsqueeze(0), memories], dim=0)
        full_context = self.transformer(sequence)
        
        # Process emotional context through attention
        emotion_context, _ = self.emotion_attention(
            current_input.unsqueeze(0),
            emotion_memories,
            emotion_memories
        )
        
        # Integrate contexts
        combined_context = self.context_integrator(
            torch.cat([
                full_context[0],
                emotion_context.squeeze(0)
            ], dim=0)
        )
        
        return combined_context, emotion_context.squeeze(0)


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
            current_state: Current input state [batch_size, embedding_dim]
            weighted_memories: Weighted sum of memories [batch_size, embedding_dim]
        
        Returns:
            Novelty scores [batch_size]
        """
        return (current_state - weighted_memories).norm(dim=-1)
