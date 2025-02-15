Emotional Embedding Examples
=======================

This document provides practical examples of using the Emotional Embedding Space (EES).

Basic Usage
----------

Here's a simple example of creating and using the EES:

.. code-block:: python

    import torch
    from nivalde.emotional_embedding import EmotionalEmbeddingSpace
    
    # Initialize the EES
    ees = EmotionalEmbeddingSpace(dim=768, memory_size=512)
    
    # Create an initial emotional state
    initial_state = torch.randn(1, 768)
    
    # Define time points for evolution
    t_span = torch.linspace(0, 1.0, 100)
    
    # Evolve the emotional state
    trajectory, transitions = ees(initial_state, t_span)

Hysteretic Memory Integration
---------------------------

Example showing how memory affects emotional state evolution:

.. code-block:: python

    from nivalde.memory import HystereticGate
    
    # Create a hysteretic gate
    gate = HystereticGate(in_features=768, persistence_factor=0.8)
    
    # Previous emotional and memory states
    prev_emotion = torch.randn(1, 768)
    memory_state = torch.randn(1, 768)
    
    # Compute gate values
    gate_values = gate(prev_emotion, memory_state)
    
    # Update emotional state with memory influence
    new_emotion = gate_values * prev_emotion + (1 - gate_values) * memory_state

Phase Transition Detection
------------------------

Example of detecting emotional phase transitions:

.. code-block:: python

    from nivalde.analysis import PhaseTransitionDetector
    
    # Create a phase transition detector
    detector = PhaseTransitionDetector(
        threshold=0.8,
        window_size=20
    )
    
    # Generate a sample trajectory
    trajectory = torch.randn(100, 1, 768)  # [time, batch, features]
    
    # Detect phase transitions
    transitions = detector(trajectory)
    
    # Print transition points
    for t, score in transitions:
        print(f"Transition at t={t:.2f} with score={score:.3f}")

Visualization
------------

Example of visualizing emotional trajectories:

.. code-block:: python

    import matplotlib.pyplot as plt
    from nivalde.visualization import plot_emotional_trajectory
    
    # Generate trajectory data
    time_points = torch.linspace(0, 1.0, 100)
    trajectory = ees(initial_state, time_points)[0]
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plot_emotional_trajectory(
        trajectory=trajectory,
        time_points=time_points,
        transitions=transitions,
        title="Emotional State Evolution"
    )
    plt.show()

Advanced Usage
------------

Example combining multiple components:

.. code-block:: python

    from nivalde.therapy import TherapyManifold
    from nivalde.memory import Transformer2Memory
    
    # Initialize components
    ees = EmotionalEmbeddingSpace(dim=768)
    memory = Transformer2Memory(dim_model=768)
    manifold = TherapyManifold(dim=64)
    
    # Process input
    input_data = torch.randn(1, 768)
    memory_state = memory(input_data, None)  # Initial memory is None
    
    # Evolve emotional state
    trajectory, transitions = ees(input_data, torch.linspace(0, 1.0, 100))
    
    # Generate therapeutic intervention
    intervention = manifold.sample_intervention(
        current_state=trajectory[-1],  # Use final emotional state
        temperature=0.8
    )
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plot_emotional_trajectory(trajectory, title="Emotional Evolution")
    
    plt.subplot(132)
    plot_memory_state(memory_state, title="Memory State")
    
    plt.subplot(133)
    plot_manifold_sample(intervention, title="Therapeutic Intervention")
    
    plt.tight_layout()
    plt.show()
