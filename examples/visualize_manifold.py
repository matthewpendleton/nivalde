"""Example of manifold visualization."""

from src.memory.temporal_manifold import TemporalManifold
from src.memory.manifold_viz import ManifoldVisualizer
import torch

def main():
    # Create and train manifold (simplified for example)
    manifold = TemporalManifold(
        manifold_dim=32,
        emotional_dim=8,
        n_heads=4
    )
    
    # Sample expressions spanning different temporal-emotional spaces
    expressions = [
        "just now",
        "this morning",
        "yesterday during the celebration",
        "last week when everything fell apart",
        "back in those darker days",
        "during the happiest moment",
        "before I understood",
        "after I learned to trust",
        "in the beginning",
        "nowadays",
        "sometime in the future",
        "at my lowest point",
        "while healing",
        "as hope returned",
        "through the struggles"
    ]
    
    # Create visualizer
    viz = ManifoldVisualizer(manifold)
    
    # Create visualizations
    viz.plot_temporal_flow(
        expressions,
        save_path="temporal_flow.html"
    )
    
    viz.plot_uncertainty_landscape(
        expressions,
        save_path="uncertainty_landscape.html"
    )
    
    viz.plot_emotional_clusters(
        expressions,
        save_path="emotional_clusters.html"
    )
    
    viz.create_interactive_dashboard(
        expressions,
        save_path="manifold_dashboard.html"
    )

if __name__ == "__main__":
    main()
