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
    """Transformer²-based memory system for capturing nuanced emotional transitions."""
    
    def __init__(
        self,
        dim=384,  # Total embedding dimension
        num_layers=4,
        num_heads=4,  # Changed from 6 to 4 heads for better memory distinctiveness
        dropout=0.1,
        temporal_decay=0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # = 96 dimensions per head
        self.temporal_decay = temporal_decay
        
        # Initialize transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=4*dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Initialize attention mechanisms
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Initialize memory storage
        self.register_buffer('memories', torch.empty(0, dim))
        self.register_buffer('emotion_memories', torch.empty(0, dim))
        self.register_buffer('surprise_scores', torch.empty(0))
        
        # Semantic preservation network
        self.semantic_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
    
    def compute_temporal_weights(self, num_memories):
        """Compute temporal decay weights for memories."""
        if num_memories == 0:
            return torch.empty(0, device=self.memories.device)
        times = torch.arange(num_memories, device=self.memories.device, dtype=torch.float32)
        weights = torch.exp(-self.temporal_decay * (num_memories - 1 - times))
        return weights / weights.sum()
    
    def store_memory(self, context, emotion=None):
        """Store a new memory with its emotional content.
        
        Args:
            context: The full contextual embedding (dim)
            emotion: The emotional embedding (dim). If None, use context.
        """
        if emotion is None:
            emotion = context
            
        # Ensure tensors are on the correct device
        context = context.to(self.memories.device)
        emotion = emotion.to(self.memories.device)
        
        # Compute surprise if we have previous memories
        if self.memories.size(0) > 0:
            surprise = self.compute_surprise(emotion, self.emotion_memories)
            self.surprise_scores = torch.cat([self.surprise_scores, surprise.unsqueeze(0)])
        else:
            self.surprise_scores = torch.cat([self.surprise_scores, torch.tensor([0.0], device=self.memories.device)])
        
        # Store new memory
        self.memories = torch.cat([self.memories, context.unsqueeze(0)], dim=0)
        self.emotion_memories = torch.cat([self.emotion_memories, emotion.unsqueeze(0)], dim=0)
    
    def compute_surprise(self, emotion, prior_emotions):
        """Compute surprise score for a new emotional state.
        
        Args:
            emotion: Current emotional state (dim)
            prior_emotions: Tensor of previous emotional states (num_memories, dim)
        
        Returns:
            Surprise score (scalar)
        """
        # If no prior emotions, return medium surprise
        if prior_emotions is None:
            return torch.tensor(0.5).to(self.memories.device)
            
        # Extract valence vectors
        current_valence = emotion[:3]
        prior_valences = prior_emotions[:, :3]
        
        # Calculate similarity using cosine similarity
        current_norm = torch.nn.functional.normalize(current_valence.unsqueeze(0), p=2, dim=1)
        prior_norms = torch.nn.functional.normalize(prior_valences, p=2, dim=1)
        similarities = torch.matmul(current_norm, prior_norms.t())
        max_similarity = torch.max(similarities)
        
        # Find the most similar prior emotion
        most_similar_idx = torch.argmax(similarities)
        most_similar_emotion = prior_valences[most_similar_idx]
        
        # Calculate magnitude of change from most similar emotion
        diff = torch.abs(current_valence - most_similar_emotion)
        total_diff = torch.sum(diff)
        
        # Calculate dominance changes
        current_dom_idx = torch.argmax(current_valence)
        prior_dom_idx = torch.argmax(most_similar_emotion)
        dom_change = current_dom_idx != prior_dom_idx
        
        # Simple surprise calculation based on similarity
        if max_similarity > 0.9:  # Very similar to most similar emotion
            valence_surprise = 0.2
        else:
            # Base surprise on total difference
            valence_surprise = 0.3 + 0.6 * total_diff / 2.0
            
            # Additional surprise for dominance changes
            if dom_change:
                valence_surprise = max(valence_surprise, 0.8)
        
        # Ensure we stay within bounds
        valence_surprise = torch.max(torch.min(torch.tensor(valence_surprise), torch.tensor(0.9)), torch.tensor(0.1))
        
        valence_surprise = valence_surprise.to(self.memories.device)
        
        return valence_surprise
    
    def compute_semantic_preservation(self, memory1, memory2):
        """Compute semantic preservation score between two memories."""
        # Normalize memories
        memory1 = F.normalize(memory1, dim=0)
        memory2 = F.normalize(memory2, dim=0)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(memory1.unsqueeze(0), memory2.unsqueeze(0))
        
        # Scale similarity to get higher preservation for similar memories
        # Adjust scaling to be less aggressive
        preservation = torch.sigmoid(3 * (similarity - 0.7))  # Lower threshold and gentler slope
        
        return preservation
    
    def get_historical_context(self, current_input=None, k=5):
        """Get the k most recent memories and their emotional content.
        
        Args:
            current_input: Optional current input to process with context
            k: Number of recent memories to retrieve
        
        Returns:
            contexts: Recent contextual memories
            emotions: Corresponding emotional content
        """
        if self.memories.size(0) == 0:
            if current_input is not None:
                return torch.zeros_like(current_input), torch.zeros_like(current_input)
            return None, None
        
        # Get most recent k memories
        start_idx = max(0, self.memories.size(0) - k)
        recent_contexts = self.memories[start_idx:]
        recent_emotions = self.emotion_memories[start_idx:]
        
        if current_input is not None:
            # Process through transformer
            sequence = torch.cat([current_input.unsqueeze(0), recent_contexts], dim=0)
            processed = self.transformer(sequence)
            
            # Apply semantic preservation
            context = processed[0]
            if recent_contexts.size(0) > 0:
                semantic_score = self.compute_semantic_preservation(
                    context,
                    recent_contexts[-1]
                )
                context = context * semantic_score + recent_contexts[-1] * (1 - semantic_score)
            
            emotion_context = recent_emotions[-1]
            return context, emotion_context
            
        return recent_contexts, recent_emotions
    
    def process_with_memory(self, input_tensor):
        """Process input using stored memories as context.
        
        Args:
            input_tensor: Input to process (dim)
        
        Returns:
            Processed tensor incorporating memory context
        """
        if self.memories.size(0) == 0:
            return self.transformer(input_tensor.unsqueeze(0)).squeeze(0)
        
        # Combine input with memories
        combined = torch.cat([self.memories, input_tensor.unsqueeze(0)], dim=0)
        
        # Process through transformer
        output = self.transformer(combined)
        
        # Apply semantic preservation
        processed = output[-1]
        if self.memories.size(0) > 0:
            semantic_score = self.compute_semantic_preservation(
                processed,
                self.memories[-1]
            )
            processed = processed * semantic_score + self.memories[-1] * (1 - semantic_score)
        
        return processed


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
