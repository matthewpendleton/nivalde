import torch
import pytest
from src.memory.memory_system import Transformer2Memory
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

def test_memory_storage_and_retrieval():
    """Test basic memory storage and retrieval functionality."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=6
    )
    
    # Create test memories
    memories = [
        torch.randn(384),  # Random memory
        torch.randn(384),  # Another random memory
        torch.randn(384)   # Third random memory
    ]
    
    # Store memories
    for memory in memories:
        memory_system.store_memory(memory)
    
    # Test retrieval
    current_input = torch.randn(384)
    context, emotion_context = memory_system.get_historical_context(current_input)
    
    assert context.shape == torch.Size([384]), "Context shape mismatch"
    assert emotion_context.shape == torch.Size([384]), "Emotion context shape mismatch"
    assert len(memory_system.memories) == 3, "Memory count mismatch"
    assert len(memory_system.emotion_memories) == 3, "Emotion memory count mismatch"

def test_emotional_and_contextual_storage():
    """Test storage of both emotional and contextual information."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=6
    )
    
    # Create test data with separate context and emotional content
    context = torch.randn(384)  # Full context
    emotion = torch.zeros(384)  # Neutral emotion
    emotion[0:3] = torch.tensor([0.8, 0.1, 0.1])  # Strong positive emotion
    
    # Store memory with both aspects
    memory_system.store_memory(context, emotion)
    
    # Verify storage
    assert len(memory_system.memories) == 1, "Context memory not stored"
    assert len(memory_system.emotion_memories) == 1, "Emotion memory not stored"
    
    # Verify the stored content
    stored_context = memory_system.memories[0]
    stored_emotion = memory_system.emotion_memories[0]
    
    assert torch.allclose(stored_context, context), "Context not stored correctly"
    assert torch.allclose(stored_emotion, emotion), "Emotion not stored correctly"

def test_surprise_computation():
    """Test surprise computation with emotional content."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=4
    )
    
    # Create baseline neutral state
    neutral_context = torch.randn(384)
    neutral_emotion = torch.zeros(384)
    neutral_emotion[0:3] = torch.tensor([0.33, 0.33, 0.34])  # Neutral emotion
    
    # Store neutral state
    memory_system.store_memory(neutral_context, neutral_emotion)
    
    # Create extreme emotional state
    extreme_context = torch.randn(384)
    extreme_emotion = torch.zeros(384)
    extreme_emotion[0:3] = torch.tensor([0.9, 0.05, 0.05])  # Very positive
    
    # Compute surprise
    surprise = memory_system.compute_surprise(
        extreme_emotion,
        memory_system.emotion_memories
    )
    
    # High emotional change should trigger high surprise
    assert surprise.item() > 0.7, "Extreme emotional change should trigger high surprise"
    
    # Reset memory system for testing similar emotions
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=4
    )
    
    # Store extreme state as baseline
    memory_system.store_memory(extreme_context, extreme_emotion)
    
    # Create similar emotional state
    similar_emotion = torch.zeros(384)
    similar_emotion[0:3] = torch.tensor([0.85, 0.1, 0.05])  # Still very positive
    
    # Compute surprise for similar state
    surprise = memory_system.compute_surprise(
        similar_emotion,
        memory_system.emotion_memories
    )
    
    # Similar emotion should have lower surprise
    assert surprise.item() < 0.3, "Similar emotion should have lower surprise"

def test_temporal_effects():
    """Test temporal effects in memory processing."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=6
    )
    
    # Create sequence of memories with both context and emotion
    base_context = torch.randn(384)
    base_emotion = torch.zeros(384)
    base_emotion[0:3] = torch.tensor([0.5, 0.3, 0.2])
    
    for i in range(5):
        context = base_context + 0.1 * i * torch.randn(384)
        emotion = torch.zeros(384)
        emotion[0:3] = torch.tensor([0.5 - 0.1*i, 0.3, 0.2 + 0.1*i])
        memory_system.store_memory(context, emotion)
    
    # Get temporal weights
    weights = memory_system.compute_temporal_weights(5)
    
    # Get historical context
    current_context = base_context + 0.5 * torch.randn(384)
    context, emotion_context = memory_system.get_historical_context(current_context)
    
    print("\nTemporal Effects Test:")
    print(f"Number of memories: {len(memory_system.memories)}")
    print(f"Context correlation: {torch.corrcoef(torch.stack([base_context, context]))[0,1]:.4f}")
    
    assert len(weights) == 5, "Incorrect number of temporal weights"
    assert weights[4] > weights[0], "Recent memories should have higher weights"
    assert context.shape == torch.Size([384]), "Context shape mismatch"
    assert emotion_context.shape == torch.Size([384]), "Emotion context shape mismatch"

def test_integration():
    """Test integration of emotional and contextual processing."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=6
    )
    
    # Create conversation sequence
    contexts = []
    emotions = []
    
    # Happy context
    contexts.append(torch.randn(384))  # "Had a great day at work!"
    emotion = torch.zeros(384)
    emotion[0:3] = torch.tensor([0.8, 0.1, 0.1])  # Very positive
    emotions.append(emotion)
    
    # Neutral context
    contexts.append(torch.randn(384))  # "Tell me more about your day."
    emotion = torch.zeros(384)
    emotion[0:3] = torch.tensor([0.33, 0.34, 0.33])  # Neutral
    emotions.append(emotion)
    
    # Sad context
    contexts.append(torch.randn(384))  # "But then I got some bad news..."
    emotion = torch.zeros(384)
    emotion[0:3] = torch.tensor([0.1, 0.1, 0.8])  # Very negative
    emotions.append(emotion)
    
    # Store sequence
    for context, emotion in zip(contexts, emotions):
        memory_system.store_memory(context, emotion)
    
    # Get integrated context
    current_input = torch.randn(384)
    context, emotion_context = memory_system.get_historical_context(current_input)
    
    print("\nIntegration Test:")
    print(f"Number of memories: {len(memory_system.memories)}")
    print(f"Context shape: {context.shape}")
    print(f"Emotion context shape: {emotion_context.shape}")
    
    assert context.shape == torch.Size([384]), "Context shape mismatch"
    assert emotion_context.shape == torch.Size([384]), "Emotion context shape mismatch"
    assert len(memory_system.memories) == len(memory_system.emotion_memories), "Memory count mismatch"

def test_emotional_transition_detection():
    """Test Transformer²'s ability to detect significant emotional shifts
    while maintaining context."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=4
    )
    
    # Create conversation with emotional transition
    # Context: "I was feeling great about my presentation..."
    context1 = torch.randn(384)
    emotion1 = torch.zeros(384)
    emotion1[0:3] = torch.tensor([0.8, 0.1, 0.1])  # Very positive
    
    # Context: "but then I completely froze up..."
    context2 = torch.randn(384)
    emotion2 = torch.zeros(384)
    emotion2[0:3] = torch.tensor([0.1, 0.1, 0.8])  # Very negative
    
    # Store sequence
    memory_system.store_memory(context1, emotion1)
    surprise = memory_system.compute_surprise(
        emotion2,
        memory_system.emotion_memories
    )
    
    # Significant emotional shift should trigger high surprise
    assert surprise.item() > 0.8, "Major emotional transition should trigger high surprise"
    
    # Store the negative state
    memory_system.store_memory(context2, emotion2)
    
    # Process current input with historical context
    current_input = torch.randn(384)
    context, emotion = memory_system.get_historical_context(current_input)
    
    # Context should reflect both emotional states
    assert context is not None
    assert emotion is not None
    assert torch.allclose(emotion, emotion2, atol=1e-6)

def test_patient_history_persistence():
    """Test that the system maintains accurate patient history."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=4
    )
    
    # Create a sequence of emotional states
    contexts = []
    emotions = []
    
    # Entry 1: Initial state
    contexts.append(torch.randn(384))
    emotion = torch.zeros(384)
    emotion[0:3] = torch.tensor([0.4, 0.3, 0.3])
    emotions.append(emotion)
    
    # Entry 2: Slight improvement
    contexts.append(torch.randn(384))
    emotion = torch.zeros(384)
    emotion[0:3] = torch.tensor([0.5, 0.3, 0.2])
    emotions.append(emotion)
    
    # Entry 3: Significant improvement
    contexts.append(torch.randn(384))
    emotion = torch.zeros(384)
    emotion[0:3] = torch.tensor([0.7, 0.2, 0.1])
    emotions.append(emotion)
    
    # Store sequence
    for i in range(len(contexts)):
        memory_system.store_memory(contexts[i], emotions[i])
    
    # Verify history
    stored_contexts, stored_emotions = memory_system.get_historical_context()
    
    assert stored_contexts.size(0) == len(contexts), f"Expected {len(contexts)} memories, got {stored_contexts.size(0)}"
    
    # Check each entry
    for i in range(len(contexts)):
        assert torch.allclose(stored_emotions[i, :3], emotions[i][:3], atol=1e-6), f"Emotion mismatch at entry {i}"
        
        # Allow for some variation in context due to semantic preservation
        context_similarity = F.cosine_similarity(stored_contexts[i], contexts[i], dim=0)
        assert context_similarity > 0.9, f"Context similarity too low at entry {i}"

def test_memory_influence():
    """Test how previous memories influence processing of new input."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=6
    )
    
    # Create base context and emotion vectors
    base_context = torch.randn(384)
    base_emotion = torch.zeros(384)
    
    # First memory: Severe anxiety
    anxiety_emotion = base_emotion.clone()
    anxiety_emotion[0:3] = torch.tensor([0.1, 0.8, 0.9])  # Very anxious
    memory_system.store_memory(base_context, anxiety_emotion)
    
    # Second memory: Some improvement
    improvement_emotion = base_emotion.clone()
    improvement_emotion[0:3] = torch.tensor([0.4, 0.5, 0.4])  # Moderately better
    memory_system.store_memory(base_context, improvement_emotion)
    
    print("\nStored Emotional Progression:")
    print("--------------------------")
    print("Initial State: Very anxious [0.1, 0.8, 0.9]")
    print("Second State: Improved [0.4, 0.5, 0.4]")
    
    # Create test input
    test_input = torch.randn(384)  # Base input embedding
    
    print("\nTesting Memory Influence:")
    print("----------------------")
    
    # 1. Process input with no memory (control)
    memory_system_empty = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=6
    )
    context_no_memory, emotion_no_memory = memory_system_empty.get_historical_context(test_input)
    
    # 2. Process same input with anxiety + improvement history
    context_with_memory, emotion_with_memory = memory_system.get_historical_context(test_input)
    
    # Compare how memories influenced processing
    context_difference = torch.norm(context_with_memory - context_no_memory)
    print(f"\nContext Processing Difference (L2): {context_difference.item():.4f}")
    
    # Show emotional progression
    print("\nEmotional Context Evolution:")
    print(f"1. Initial Anxiety: {anxiety_emotion[0:3].tolist()}")
    print(f"2. Some Improvement: {improvement_emotion[0:3].tolist()}")
    print(f"3. Current State: {emotion_with_memory[0:3].tolist()}")
    
    # Verify memory influence
    assert context_difference > 0.1, "Memory should significantly influence processing"
    assert torch.allclose(emotion_with_memory, improvement_emotion), "Should maintain latest emotional state"

def test_embedding_precision():
    """Test if 384 dimensions are sufficient for precise memory storage."""
    
    # Test different embedding dimensions
    dimensions = [96, 192, 384, 768]
    
    results = {}
    for dim in dimensions:
        memory_system = Transformer2Memory(
            dim=dim,
            num_layers=4,
            num_heads=6
        )
        
        # Create a set of subtle variations in emotional state
        base_emotion = torch.zeros(dim)
        base_context = torch.randn(dim)
        
        subtle_variations = [
            # Slight variations in anxiety/stress
            ([0.30, 0.45, 0.40], "Mild anxiety"),
            ([0.32, 0.45, 0.40], "Slightly higher anxiety"),
            ([0.30, 0.47, 0.40], "More physical stress"),
            ([0.30, 0.45, 0.42], "More emotional tension"),
            
            # Subtle mood changes
            ([0.55, 0.30, 0.25], "Generally positive"),
            ([0.57, 0.30, 0.25], "Slightly more positive"),
            ([0.55, 0.28, 0.25], "Less physical tension"),
            ([0.55, 0.30, 0.23], "Less emotional weight")
        ]
        
        print(f"\nTesting {dim} dimensions:")
        print("------------------------")
        
        # Store each variation
        for values, description in subtle_variations:
            emotion = base_emotion.clone()
            emotion[0:3] = torch.tensor(values)
            memory_system.store_memory(base_context, emotion)
        
        # Test discrimination between similar states
        cosine_similarities = []
        euclidean_distances = []
        
        memories = memory_system.emotion_memories
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                mem1 = memories[i]
                mem2 = memories[j]
                
                # Compute similarity metrics
                cosine_sim = torch.nn.functional.cosine_similarity(mem1.unsqueeze(0), mem2.unsqueeze(0))
                euclidean_dist = torch.norm(mem1 - mem2)
                
                cosine_similarities.append(cosine_sim.item())
                euclidean_distances.append(euclidean_dist.item())
                
                # Print comparison for very similar states
                if cosine_sim > 0.99:
                    print(f"\nVery similar states detected:")
                    print(f"State 1: {subtle_variations[i][0]}")
                    print(f"State 2: {subtle_variations[j][0]}")
                    print(f"Cosine Similarity: {cosine_sim.item():.4f}")
                    print(f"Euclidean Distance: {euclidean_dist.item():.4f}")
        
        # Compute statistics
        avg_cosine = sum(cosine_similarities) / len(cosine_similarities)
        avg_distance = sum(euclidean_distances) / len(euclidean_distances)
        min_distance = min(euclidean_distances)
        
        results[dim] = {
            'avg_cosine': avg_cosine,
            'avg_distance': avg_distance,
            'min_distance': min_distance
        }
        
        print(f"\nDimension {dim} Statistics:")
        print(f"Average Cosine Similarity: {avg_cosine:.4f}")
        print(f"Average Euclidean Distance: {avg_distance:.4f}")
        print(f"Minimum Euclidean Distance: {min_distance:.4f}")
    
    # Compare dimensionality results
    print("\nDimensionality Comparison:")
    print("-------------------------")
    for dim, stats in results.items():
        print(f"\n{dim} dimensions:")
        print(f"- Avg Similarity: {stats['avg_cosine']:.4f}")
        print(f"- Avg Distance: {stats['avg_distance']:.4f}")
        print(f"- Min Distance: {stats['min_distance']:.4f}")
    
    # Verify sufficient precision
    final_dim = 384
    assert results[final_dim]['min_distance'] > 0.01, "Should maintain distinction between subtle variations"
    assert results[final_dim]['avg_cosine'] < 0.99, "Should not over-collapse similar states"

def test_full_memory_precision():
    """Test memory precision with focus on semantic preservation."""
    memory_system = Transformer2Memory(
        dim=384,
        num_layers=4,
        num_heads=4
    )
    
    # Create a sequence of related memories
    base_memory = torch.randn(384)
    
    # Store base memory
    memory_system.store_memory(base_memory)
    
    # Create and store related memories with controlled variation
    num_memories = 5
    semantic_scores = []
    
    for i in range(num_memories):
        # Create related memory with controlled noise
        variation = 0.1 * torch.randn(384)
        related_memory = base_memory + variation
        
        # Store memory
        memory_system.store_memory(related_memory)
        
        # Compute semantic preservation
        score = memory_system.compute_semantic_preservation(
            related_memory,
            base_memory
        )
        semantic_scores.append(score.item())
    
    # Check semantic preservation
    avg_semantic_score = sum(semantic_scores) / len(semantic_scores)
    assert 0.6 < avg_semantic_score < 0.9, "Should balance similarity and distinctness"
    
    # Test retrieval with semantic preservation
    query = base_memory + 0.2 * torch.randn(384)
    context, _ = memory_system.get_historical_context(query)
    
    # Verify semantic relationships are preserved
    similarity = F.cosine_similarity(context, base_memory, dim=0)
    assert similarity > 0.9, "Should preserve semantic relationships"

def test_attention_head_configurations():
    """Test different attention head configurations with fixed total dimension."""
    
    total_dim = 384  # Keep total dimensions fixed
    head_configs = [
        (2, 192),   # 2 heads, 192 dims per head
        (3, 128),   # 3 heads, 128 dims per head
        (4, 96),    # 4 heads, 96 dims per head
        (6, 64),    # 6 heads, 64 dims per head (original)
    ]
    
    # Create complex contextual scenarios
    scenarios = [
        # Subtle variations in similar contexts
        (
            "Patient expresses anxiety about work deadlines, particularly about a presentation next week",
            "Patient discusses work stress, focusing on an upcoming presentation and team dynamics",
            "Similar but distinct work scenarios"
        ),
        (
            "Discussion about family relationships, focusing on communication with siblings",
            "Conversation about family dynamics, emphasizing parent-child relationships",
            "Related family contexts"
        ),
        # Complex emotional-contextual interactions
        (
            "Expression of pride in recent accomplishments while showing uncertainty about future goals",
            "Discussion of future aspirations with mix of confidence and apprehension",
            "Mixed emotional states"
        )
    ]
    
    results = {}
    for num_heads, dims_per_head in head_configs:
        print(f"\nTesting configuration: {num_heads} heads with {dims_per_head} dimensions per head")
        print("-" * 80)
        
        memory_system = Transformer2Memory(
            dim=total_dim,
            num_layers=4,
            num_heads=num_heads
        )
        
        # Store each scenario
        memory_pairs = []
        for scenario1, scenario2, _ in scenarios:
            # Create distinct but related memory embeddings
            memory1 = torch.randn(total_dim)
            # Create a related but distinct memory with controlled variation
            memory2 = memory1 + 0.2 * torch.randn(total_dim)  # Increased variation
            memory_pairs.append((memory1, memory2))
            
            # Create corresponding emotion vectors
            emotion1 = torch.zeros(total_dim)
            emotion1[:3] = torch.tensor([0.3, 0.4, 0.5])
            emotion2 = torch.zeros(total_dim)
            emotion2[:3] = torch.tensor([0.32, 0.38, 0.48])
            
            # Store memories
            memory_system.store_memory(memory1, emotion1)
            memory_system.store_memory(memory2, emotion2)
        
        # Analyze memory distinctions
        cosine_similarities = []
        euclidean_distances = []
        attention_patterns = []
        
        memories = memory_system.memories
        for i in range(0, len(memories), 2):
            mem1 = memories[i]
            mem2 = memories[i+1]
            
            # Compute metrics
            cosine_sim = torch.nn.functional.cosine_similarity(mem1.unsqueeze(0), mem2.unsqueeze(0))
            euclidean_dist = torch.norm(mem1 - mem2)
            
            # Analyze attention patterns per head
            query = mem2.unsqueeze(0).unsqueeze(0)
            key = memory_system.emotion_memories.unsqueeze(0)
            value = key  # Use same tensor for key and value
            
            attn_output, attn_weights = memory_system.emotion_attention(
                query, key, value
            )
            
            # Get attention weights for each head
            # attn_weights shape: [batch_size=1, num_heads, seq_len_q=1, seq_len_k]
            head_attention = attn_weights.squeeze(0).squeeze(1)  # Remove batch and seq_len_q dims
            attention_patterns.append(head_attention)
            
            cosine_similarities.append(cosine_sim.item())
            euclidean_distances.append(euclidean_dist.item())
            
            scenario_desc = scenarios[i//2][2]
            print(f"\nScenario: {scenario_desc}")
            print(f"Cosine Similarity: {cosine_sim.item():.4f}")
            print(f"Euclidean Distance: {euclidean_dist.item():.4f}")
            print(f"Attention Distribution:")
            for h in range(head_attention.size(0)):
                print(f"  Head {h}: {head_attention[h].mean().item():.4f}")
        
        # Compute statistics
        avg_attention_entropy = torch.mean(torch.tensor([
            -torch.sum(pattern * torch.log(pattern + 1e-10))
            for pattern in attention_patterns
        ]))
        
        results[(num_heads, dims_per_head)] = {
            'avg_cosine': sum(cosine_similarities) / len(cosine_similarities),
            'avg_distance': sum(euclidean_distances) / len(euclidean_distances),
            'min_distance': min(euclidean_distances),
            'attention_entropy': avg_attention_entropy.item(),
            'attention_patterns': attention_patterns
        }
        
        print(f"\nConfiguration Statistics:")
        print(f"Average Cosine Similarity: {results[(num_heads, dims_per_head)]['avg_cosine']:.4f}")
        print(f"Average Euclidean Distance: {results[(num_heads, dims_per_head)]['avg_distance']:.4f}")
        print(f"Minimum Euclidean Distance: {results[(num_heads, dims_per_head)]['min_distance']:.4f}")
        print(f"Attention Entropy: {results[(num_heads, dims_per_head)]['attention_entropy']:.4f}")
    
    # Compare configurations
    print("\nConfiguration Comparison:")
    print("-" * 40)
    for (heads, dims), stats in results.items():
        print(f"\n{heads} heads with {dims} dimensions per head:")
        print(f"- Avg Similarity: {stats['avg_cosine']:.4f}")
        print(f"- Avg Distance: {stats['avg_distance']:.4f}")
        print(f"- Min Distance: {stats['min_distance']:.4f}")
        print(f"- Attention Entropy: {stats['attention_entropy']:.4f}")
    
    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1]['avg_cosine'])
    print(f"\nBest Configuration: {best_config[0][0]} heads with {best_config[0][1]} dimensions per head")
    print(f"Achieved cosine similarity: {best_config[1]['avg_cosine']:.4f}")

if __name__ == "__main__":
    print("Testing Transformer² Memory System...")
    
    test_memory_storage_and_retrieval()
    test_emotional_and_contextual_storage()
    test_surprise_computation()
    test_temporal_effects()
    test_integration()
    test_emotional_transition_detection()
    test_patient_history_persistence()
    test_memory_influence()
    test_embedding_precision()
    test_full_memory_precision()
    test_attention_head_configurations()
    
    print("\nAll tests completed!")
