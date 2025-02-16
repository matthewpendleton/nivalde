import sys
import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.emotional_embedding.ees import EmotionalEmbeddingSpace
from src.memory.memory_system import Transformer2Memory
from src.input.multimodal_processor import MultimodalProcessor

def test_state_initialization():
    """Test emotional state initialization and basic properties."""
    ees = EmotionalEmbeddingSpace(dim=768)
    
    # Test with random initial state
    initial_state = torch.randn(768)
    bert_context = torch.randn(768)
    memory_context = torch.randn(768)
    
    # Forward pass
    new_state = ees(initial_state, bert_context, memory_context)
    
    # Basic assertions
    assert new_state is not None
    assert new_state.shape == (768,)
    assert torch.isfinite(new_state).all(), "State contains inf or nan values"
    
    print("\nState Initialization Test:")
    print(f"Initial state norm: {initial_state.norm():.4f}")
    print(f"New state norm: {new_state.norm():.4f}")

def test_emotional_hysteresis():
    """Test that the system exhibits emotional hysteresis (state memory)."""
    ees = EmotionalEmbeddingSpace(dim=768)
    
    # Create sequence of related states
    base_state = torch.ones(768)
    base_state = base_state / base_state.norm()
    
    # Create slightly different contexts
    contexts = [
        (torch.ones(768) + 0.1 * torch.randn(768)),  # Similar
        (torch.ones(768) + 0.3 * torch.randn(768)),  # More different
        (torch.ones(768) + 0.5 * torch.randn(768)),  # Most different
    ]
    contexts = [c / c.norm() for c in contexts]
    
    # Evolve state
    states = []
    current_state = base_state
    for bert_context in contexts:
        memory_context = torch.zeros(768)  # Zero memory for controlled test
        new_state = ees(current_state, bert_context, memory_context)
        states.append(new_state)
        current_state = new_state
    
    # Compute state changes
    state_changes = [
        torch.norm(states[i] - states[i-1]).item()
        for i in range(1, len(states))
    ]
    
    print("\nEmotional Hysteresis Test:")
    print("State changes between sequential updates:")
    for i, change in enumerate(state_changes):
        print(f"Change {i} -> {i+1}: {change:.4f}")
    
    # State should change gradually
    assert all(change < 1.0 for change in state_changes), "State changes too abrupt"

def test_context_integration():
    """Test integration of BERT and memory context."""
    ees = EmotionalEmbeddingSpace(dim=768)
    processor = MultimodalProcessor()
    memory = Transformer2Memory(dim=768)
    
    # Test emotional progression
    test_sentences = [
        "I feel completely overwhelmed and anxious.",
        "Taking deep breaths helps me calm down a bit.",
        "I'm starting to feel more centered and in control.",
        "Now I can think more clearly about the situation.",
        "I feel much more peaceful and balanced now."
    ]
    
    # Process sequence
    states = []
    current_state = None
    
    for i, sentence in enumerate(test_sentences):
        # Get BERT embedding
        bert_context = processor.process_text(sentence)
        if len(bert_context.shape) > 1:
            bert_context = bert_context.squeeze()
        
        # Get memory context
        memory_context = (
            memory.get_historical_context(bert_context)
            if i > 0 else torch.zeros_like(bert_context)
        )
        
        # Initialize or update state
        if current_state is None:
            current_state = bert_context
        
        # Update state
        new_state = ees(current_state, bert_context, memory_context)
        states.append(new_state)
        current_state = new_state
        
        # Store in memory
        memory.store_memory(bert_context)
    
    # Convert states to tensor for analysis
    states_tensor = torch.stack(states)
    
    # Compute trajectory smoothness
    velocity = torch.diff(states_tensor, dim=0)
    smoothness = torch.mean(torch.norm(velocity, dim=1)).item()
    
    print("\nContext Integration Test:")
    print(f"Number of states: {len(states)}")
    print(f"Trajectory smoothness: {smoothness:.4f}")
    
    # Compute correlations between sequential states
    corr_matrix = torch.corrcoef(states_tensor)
    print("\nCorrelation between sequential states:")
    for i in range(len(states)-1):
        print(f"Correlation {i} -> {i+1}: {corr_matrix[i,i+1]:.4f}")
    
    # Visualize emotional trajectory
    plot_emotional_trajectory(states_tensor, test_sentences)

def test_stability():
    """Test stability of emotional state updates."""
    ees = EmotionalEmbeddingSpace(dim=768)
    
    # Test with constant input
    constant_state = torch.ones(768)
    constant_state = constant_state / constant_state.norm()
    constant_context = constant_state.clone()
    
    # Multiple updates
    states = []
    current_state = constant_state
    
    for _ in range(10):
        new_state = ees(current_state, constant_context, constant_context)
        states.append(new_state)
        current_state = new_state
    
    # Compute state changes
    changes = [
        torch.norm(states[i] - states[i-1]).item()
        for i in range(1, len(states))
    ]
    
    print("\nStability Test:")
    print("State changes with constant input:")
    for i, change in enumerate(changes):
        print(f"Change {i}: {change:.4f}")
    
    # Changes should decrease
    assert all(changes[i] >= changes[i+1] for i in range(len(changes)-1)), \
        "State not stabilizing with constant input"

def plot_emotional_trajectory(states_tensor, labels):
    """Visualize emotional state trajectory using PCA."""
    # Detach tensors and move to CPU for plotting
    states_tensor = states_tensor.detach()
    
    # Perform PCA
    U, S, V = torch.pca_lowrank(states_tensor, q=2)
    trajectory_2d = torch.matmul(states_tensor, V[:, :2])
    
    # Convert to numpy for plotting
    trajectory_2d = trajectory_2d.numpy()
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot points
    plt.scatter(
        trajectory_2d[:, 0],
        trajectory_2d[:, 1],
        c=np.arange(len(trajectory_2d)),
        cmap='viridis',
        s=100
    )
    
    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(
            f"{i+1}",
            (trajectory_2d[i, 0], trajectory_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    # Connect points with arrows
    for i in range(len(trajectory_2d)-1):
        plt.arrow(
            trajectory_2d[i, 0],
            trajectory_2d[i, 1],
            trajectory_2d[i+1, 0] - trajectory_2d[i, 0],
            trajectory_2d[i+1, 1] - trajectory_2d[i, 1],
            head_width=0.1,
            head_length=0.1,
            fc='k',
            ec='k',
            alpha=0.3
        )
    
    plt.title("Emotional State Trajectory")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    
    # Save plot
    output_dir = Path(project_root) / 'tests' / 'output'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'emotional_trajectory.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Testing Emotional Embedding Continuum...")
    
    test_state_initialization()
    test_emotional_hysteresis()
    test_context_integration()
    test_stability()
    
    print("\nAll tests completed!")
