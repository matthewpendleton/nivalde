import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

class EmotionalEmbeddingSpace(nn.Module):
    def __init__(self, embedding_dim: int, n_attractors: int = 10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_attractors = n_attractors
        
        # Learnable attractors representing emotional states
        self.attractors = nn.Parameter(torch.randn(n_attractors, embedding_dim))
        
        # Hysteresis parameters
        self.hysteresis_strength = nn.Parameter(torch.ones(n_attractors))
        self.phase_transition_thresholds = nn.Parameter(torch.rand(n_attractors))
        
    def compute_attractor_influence(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute the influence of emotional attractors on the current state"""
        distances = torch.cdist(embedding.unsqueeze(0), self.attractors.unsqueeze(0))[0]
        influence = torch.softmax(-distances * self.hysteresis_strength, dim=-1)
        return influence
        
    def detect_phase_transitions(self, 
                               current_state: torch.Tensor,
                               previous_state: Optional[torch.Tensor] = None) -> List[int]:
        """Detect potential therapeutic opportunities or risks"""
        if previous_state is None:
            return []
            
        state_change = torch.norm(current_state - previous_state)
        transitions = []
        
        for i, threshold in enumerate(self.phase_transition_thresholds):
            if state_change > threshold:
                transitions.append(i)
                
        return transitions

class TransformerMemoryBlock(nn.Module):
    def __init__(self, 
                 embedding_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4):
        super().__init__()
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=4*embedding_dim
            ),
            num_layers=num_layers
        )
        
        # Memory bank
        self.memories = []
        self.memory_importance = []
        
    def update_memory(self, 
                     new_memory: torch.Tensor,
                     importance_score: float):
        """Add new memory with importance-based retention"""
        self.memories.append(new_memory)
        self.memory_importance.append(importance_score)
        
        # Prune less important memories if needed
        if len(self.memories) > 1000:  # Arbitrary limit
            indices = np.argsort(self.memory_importance)[:-1000]
            self.memories = [self.memories[i] for i in indices]
            self.memory_importance = [self.memory_importance[i] for i in indices]
            
    def retrieve_relevant_memories(self, 
                                 current_context: torch.Tensor,
                                 top_k: int = 5) -> torch.Tensor:
        """Retrieve most relevant memories for current context"""
        if not self.memories:
            return torch.zeros_like(current_context)
            
        memory_tensor = torch.stack(self.memories)
        similarities = torch.matmul(current_context, memory_tensor.T)
        _, indices = torch.topk(similarities, min(top_k, len(self.memories)))
        
        relevant_memories = memory_tensor[indices]
        return self.transformer(relevant_memories)

class EmotionalProcessor:
    def __init__(self, embedding_dim: int = 768):  
        self.embedding_space = EmotionalEmbeddingSpace(embedding_dim)
        self.memory_block = TransformerMemoryBlock(embedding_dim)
        self.previous_state = None
        
    def process(self, 
                contextual_embedding: torch.Tensor,
                store_memory: bool = True) -> Tuple[torch.Tensor, List[int]]:
        """Process contextual embedding through emotional embedding space"""
        # Get relevant memories
        memory_context = self.memory_block.retrieve_relevant_memories(contextual_embedding)
        
        # Combine current context with memory
        combined_state = contextual_embedding + 0.5 * memory_context.mean(0)
        
        # Process through emotional space
        attractor_influence = self.embedding_space.compute_attractor_influence(combined_state)
        current_state = combined_state * (1 + attractor_influence.mean())
        
        # Detect phase transitions
        transitions = self.embedding_space.detect_phase_transitions(
            current_state, self.previous_state
        )
        
        # Update memory if needed
        if store_memory:
            importance = attractor_influence.max().item()
            self.memory_block.update_memory(current_state, importance)
        
        self.previous_state = current_state
        return current_state, transitions
