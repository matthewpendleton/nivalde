import sys
import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.memory.memory_system import Transformer2Memory
from src.input.multimodal_processor import MultimodalProcessor

def test_memory_storage_and_retrieval():
    """Test basic memory storage and retrieval functionality."""
    memory_system = Transformer2Memory(dim=768)
    
    # Create some test memories
    test_memories = [
        torch.randn(768),  # Random memory
        torch.ones(768),   # All ones
        torch.zeros(768),  # All zeros
        torch.randn(768)   # Another random memory
    ]
    
    # Store memories
    for memory in test_memories:
        memory_system.store_memory(memory)
    
    # Test retrieval with current input
    current_input = torch.randn(768)
    historical_context = memory_system.get_historical_context(current_input)
    
    # Basic assertions
    assert historical_context is not None
    assert historical_context.shape == (768,)
    
    print("\nMemory Storage and Retrieval Test:")
    print(f"Number of stored memories: {len(memory_system.memories)}")
    print(f"Historical context shape: {historical_context.shape}")

def test_surprise_computation():
    """Test surprise/novelty scoring mechanism."""
    memory_system = Transformer2Memory(dim=768)
    
    # Create test cases with controlled differences
    base_memory = torch.ones(768)
    base_memory = base_memory / base_memory.norm()  # Normalize base vector
    
    # Create similar memory (small angle difference)
    similar_memory = base_memory + 0.1 * torch.randn(768)
    similar_memory = similar_memory / similar_memory.norm()
    
    # Create different memory (orthogonal)
    different_memory = torch.randn(768)
    different_memory = different_memory - torch.dot(different_memory, base_memory) * base_memory
    different_memory = different_memory / different_memory.norm()
    
    # Store base memory
    memory_system.store_memory(base_memory)
    
    # Compute surprise scores
    similar_score = memory_system.compute_surprise(similar_memory, torch.stack([base_memory]))
    different_score = memory_system.compute_surprise(different_memory, torch.stack([base_memory]))
    
    print("\nSurprise Computation Test:")
    print(f"Surprise score for similar memory: {similar_score:.4f}")
    print(f"Surprise score for different memory: {different_score:.4f}")
    print(f"Cosine similarity for similar memory: {torch.dot(base_memory, similar_memory):.4f}")
    print(f"Cosine similarity for different memory: {torch.dot(base_memory, different_memory):.4f}")
    
    # Different memory should be more surprising (less similar) than similar memory
    assert different_score > similar_score, "Different memory should have higher surprise score"

def test_temporal_effects():
    """Test how temporal distance affects memory retrieval."""
    memory_system = Transformer2Memory(dim=768)
    
    # Create sequence of related memories
    base = torch.randn(768)
    memories = [
        base + 0.1 * torch.randn(768),  # t-3: slightly different
        base + 0.2 * torch.randn(768),  # t-2: more different
        base + 0.3 * torch.randn(768),  # t-1: most different
    ]
    
    # Store memories in sequence
    for memory in memories:
        memory_system.store_memory(memory)
    
    # Get historical context
    context = memory_system.get_historical_context(base)
    
    print("\nTemporal Effects Test:")
    print(f"Base memory correlation with context: {torch.corrcoef(torch.stack([base, context]))[0,1]:.4f}")

def test_integration():
    """Test integration with BERT embeddings."""
    memory_system = Transformer2Memory(dim=768)
    processor = MultimodalProcessor()
    
    # Test sentences with emotional progression
    test_sentences = [
        "I feel quite happy today.",
        "The happiness is mixed with anticipation.",
        "Now I'm starting to feel a bit anxious.",
        "The anxiety is overwhelming.",
        "I'm trying to stay calm and centered."
    ]
    
    # Process sentences and store memories
    embeddings = []
    for sentence in test_sentences:
        # Get BERT embedding
        embedding = processor.process_text(sentence)
        
        # Ensure embedding is properly shaped
        if len(embedding.shape) > 1:
            embedding = embedding.squeeze()
        
        embeddings.append(embedding)
        
        # Store in memory system
        memory_system.store_memory(embedding)
    
    # Get historical context for last embedding
    context = memory_system.get_historical_context(embeddings[-1])
    
    print("\nIntegration Test:")
    print(f"Number of processed memories: {len(memory_system.memories)}")
    print(f"Context shape: {context.shape}")
    
    # Compute correlations between sequential memories
    embeddings_tensor = torch.stack(embeddings)
    corr_matrix = torch.corrcoef(embeddings_tensor)
    
    print("\nCorrelation matrix between sequential memories:")
    for i in range(len(test_sentences)-1):
        print(f"Correlation {i} -> {i+1}: {corr_matrix[i,i+1]:.4f}")

if __name__ == "__main__":
    print("Testing Transformer² Memory System...")
    
    test_memory_storage_and_retrieval()
    test_surprise_computation()
    test_temporal_effects()
    test_integration()
    
    print("\nAll tests completed!")
