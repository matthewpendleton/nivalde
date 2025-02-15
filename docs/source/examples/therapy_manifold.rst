Therapy Manifold Examples
=====================

This document provides practical examples of using the Therapy Manifold.

Basic Usage
----------

Here's a simple example of creating and using the Therapy Manifold:

.. code-block:: python

    import torch
    from nivalde.therapy import TherapyManifold
    
    # Initialize the manifold
    manifold = TherapyManifold(dim=64, n_steps=100)
    
    # Current therapeutic state
    current_state = torch.randn(1, 64)
    
    # Sample intervention
    intervention = manifold.sample_intervention(
        current_state=current_state,
        temperature=1.0
    )

Riemannian Geometry Operations
---------------------------

Example of computing geodesics and parallel transport:

.. code-block:: python

    from nivalde.geometry import RiemannianMetric
    
    # Create metric
    metric = RiemannianMetric(dim=64)
    
    # Two points on the manifold
    x = torch.randn(1, 64)
    y = torch.randn(1, 64)
    
    # Compute geodesic
    t = torch.linspace(0, 1, 10)
    geodesic = metric.geodesic(x, y, t)
    
    # Parallel transport a vector
    v = torch.randn(1, 64)
    transported = metric.parallel_transport(x, y, v)

Intervention Optimization
----------------------

Example of optimizing interventions on the manifold:

.. code-block:: python

    from nivalde.optimization import ManifoldOptimizer
    
    # Create optimizer
    optimizer = ManifoldOptimizer(
        manifold=manifold,
        learning_rate=0.01
    )
    
    # Define objective function
    def objective(x):
        return manifold.potential(x)
    
    # Optimize intervention
    initial_point = torch.randn(1, 64)
    optimal_intervention = optimizer.optimize(
        objective=objective,
        initial_point=initial_point,
        n_steps=100
    )

Visualization
------------

Example of visualizing the manifold and interventions:

.. code-block:: python

    import matplotlib.pyplot as plt
    from nivalde.visualization import plot_manifold
    
    # Generate sample points
    points = torch.randn(100, 64)
    
    # Project to 2D for visualization
    projected_points = manifold.project_to_2d(points)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plot_manifold(
        points=projected_points,
        metric=metric,
        title="Therapy Manifold"
    )
    plt.show()

Advanced Usage
------------

Example combining manifold with emotional embedding:

.. code-block:: python

    from nivalde.emotional_embedding import EmotionalEmbeddingSpace
    from nivalde.therapy import ResponseGenerator
    
    # Initialize components
    ees = EmotionalEmbeddingSpace(dim=768)
    manifold = TherapyManifold(dim=64)
    generator = ResponseGenerator(
        manifold=manifold,
        embedding_dim=768,
        manifold_dim=64
    )
    
    # Process emotional state
    emotional_state = torch.randn(1, 768)
    trajectory, _ = ees(emotional_state, torch.linspace(0, 1.0, 100))
    
    # Map to manifold and generate response
    manifold_point = generator.emotional_to_manifold(trajectory[-1])
    intervention = manifold.sample_intervention(
        current_state=manifold_point,
        temperature=0.8
    )
    response = generator.manifold_to_response(intervention)
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plot_emotional_trajectory(trajectory, title="Emotional State")
    
    plt.subplot(132)
    plot_manifold_point(manifold_point, title="Manifold Mapping")
    
    plt.subplot(133)
    plot_intervention(intervention, response, title="Generated Response")
    
    plt.tight_layout()
    plt.show()
