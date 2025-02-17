import sys
import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

def test_emotional_memory_persistence():
    """Test how well the Transformer² maintains emotional context over time."""
    memory_system = Transformer2Memory(
        dim=384,          # Emotional context dimension
        num_layers=4,     # Fewer layers for faster processing
        num_heads=6       # 64-dim per head for emotional aspects
    )
    
    # Create sequence of related emotional states
    joy = torch.tensor([0.8, 0.1, 0.0] * 128)  # High positive valence
    contentment = torch.tensor([0.6, 0.2, 0.1] * 128)  # Moderate positive
    neutral = torch.tensor([0.3, 0.3, 0.3] * 128)  # Neutral state
    concern = torch.tensor([0.2, 0.5, 0.4] * 128)  # Slight negative
    anxiety = torch.tensor([0.1, 0.7, 0.8] * 128)  # High negative
    
    # Store emotional sequence
    emotional_sequence = [joy, contentment, neutral, concern, anxiety]
    for emotion in emotional_sequence:
        memory_system.store_memory(emotion)
    
    # Create sequence tensor for transformer
    sequence = torch.stack(memory_system.memories)
    
    # Process through transformer
    processed = memory_system.transformer(sequence)
    
    # Test retrieval with similar emotional state
    test_joy = torch.tensor([0.75, 0.15, 0.05] * 128).unsqueeze(0)  # Add batch dim
    
    # Get attention weights for test state
    with torch.no_grad():
        # Self-attention with test state
        test_sequence = torch.cat([sequence, test_joy])
        processed_with_test = memory_system.transformer(test_sequence)
        final_state = processed_with_test[-1]  # Get transformed test state
        
        # Compare similarities
        joy_similarity = F.cosine_similarity(final_state, processed[0], dim=0)
        anxiety_similarity = F.cosine_similarity(final_state, processed[-1], dim=0)
    
    # Context should be more similar to joy than anxiety
    assert joy_similarity > anxiety_similarity
    
    print("\nEmotional Memory Test:")
    print(f"Joy similarity: {joy_similarity:.3f}")
    print(f"Anxiety similarity: {anxiety_similarity:.3f}")

def test_emotional_transition_detection():
    """Test Transformer²'s ability to detect significant emotional shifts."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=6
    )
    
    # Create gradual transition
    gradual_shift = [
        torch.tensor([0.8 - 0.1*i, 0.1 + 0.1*i, 0.1] * 128)
        for i in range(8)
    ]
    
    # Store and get surprise scores for gradual transition
    surprise_scores_gradual = []
    for i, state in enumerate(gradual_shift):
        # Skip first state for surprise computation
        if i > 0:  # Only compute surprise after first memory
            surprise = memory_system.compute_surprise(
                state,
                torch.stack(memory_system.memories)
            )
            surprise_scores_gradual.append(surprise)
        memory_system.store_memory(state)
    
    print("\nGradual Transition Scores:")
    for i, score in enumerate(surprise_scores_gradual):
        print(f"Step {i+1}: {score:.3f}")
    
    # Reset memory system
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=6
    )
    
    # Create sudden transition
    sudden_shift = [
        torch.tensor([0.8, 0.1, 0.1] * 128),  # Very positive
        torch.tensor([0.1, 0.1, 0.8] * 128)   # Very negative
    ]
    
    # Store and get surprise scores for sudden transition
    surprise_scores_sudden = []
    for i, state in enumerate(sudden_shift):
        # Skip first state for surprise computation
        if i > 0:  # Only compute surprise after first memory
            surprise = memory_system.compute_surprise(
                state,
                torch.stack(memory_system.memories)
            )
            surprise_scores_sudden.append(surprise)
        memory_system.store_memory(state)
    
    print("\nSudden Transition Scores:")
    for i, score in enumerate(surprise_scores_sudden):
        print(f"Step {i+1}: {score:.3f}")
    
    # Get maximum non-initial surprise scores
    max_surprise_gradual = max(surprise_scores_gradual) if surprise_scores_gradual else 0
    max_surprise_sudden = max(surprise_scores_sudden) if surprise_scores_sudden else 0
    
    print(f"\nMax Surprise Scores:")
    print(f"Gradual: {max_surprise_gradual:.3f}")
    print(f"Sudden:  {max_surprise_sudden:.3f}")
    
    # Sudden transition should show higher surprise
    assert max_surprise_sudden > max_surprise_gradual, \
        f"Sudden transition ({max_surprise_sudden:.3f}) should be more surprising than gradual ({max_surprise_gradual:.3f})"
    
    # Visualize surprise scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(surprise_scores_gradual) + 1), 
             surprise_scores_gradual, 
             'b-o', 
             label='Gradual Transition')
    plt.plot(len(surprise_scores_gradual) + 1, 
             surprise_scores_sudden[0], 
             'r-o', 
             label='Sudden Transition')
    plt.xlabel('Memory Step')
    plt.ylabel('Surprise Score')
    plt.title('Emotional Transition Detection')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Testing Transformer² Memory System...")
    
    test_memory_storage_and_retrieval()
    test_surprise_computation()
    test_temporal_effects()
    test_integration()
    test_emotional_memory_persistence()
    test_emotional_transition_detection()
    
    print("\nAll tests completed!")
