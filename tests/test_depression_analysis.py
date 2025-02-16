"""Test suite for analyzing depression vs non-depression emotional patterns."""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import silhouette_score

from src.emotional_embedding.ees import EmotionalEmbeddingSpace
from src.memory.memory_system import TransformerMemorySystem
from tests.test_data.depression_scenarios import (
    DEPRESSION_SCENARIOS,
    NON_DEPRESSION_SCENARIOS,
    EMOTIONAL_MARKERS
)

def analyze_emotional_trajectories():
    """Analyze emotional trajectories for depression vs non-depression scenarios."""
    # Initialize models
    ees = EmotionalEmbeddingSpace(dim=768)
    memory_system = TransformerMemorySystem(embedding_dim=768)
    
    # Process all scenarios
    depression_embeddings = []
    non_depression_embeddings = []
    
    # Process depression scenarios
    for scenario in DEPRESSION_SCENARIOS:
        scenario_embeddings = []
        memory_context = torch.zeros(1, 768)  # Initial memory context
        
        for utterance in scenario["dialogue"]:
            # Simulate BERT embeddings (replace with actual BERT in production)
            current_embedding = torch.randn(1, 768)  # Placeholder
            
            # Get emotional state
            state = ees(
                current_state=current_embedding,
                bert_context=current_embedding,
                memory_context=memory_context
            )
            
            # Update memory
            memory_context = memory_system(
                current_input=current_embedding,
                previous_memory=memory_context
            )
            
            scenario_embeddings.append(state.detach())
            
        depression_embeddings.extend(scenario_embeddings)
    
    # Process non-depression scenarios
    for scenario in NON_DEPRESSION_SCENARIOS:
        scenario_embeddings = []
        memory_context = torch.zeros(1, 768)  # Initial memory context
        
        for utterance in scenario["dialogue"]:
            # Simulate BERT embeddings (replace with actual BERT in production)
            current_embedding = torch.randn(1, 768)  # Placeholder
            
            # Get emotional state
            state = ees(
                current_state=current_embedding,
                bert_context=current_embedding,
                memory_context=memory_context
            )
            
            # Update memory
            memory_context = memory_system(
                current_input=current_embedding,
                previous_memory=memory_context
            )
            
            scenario_embeddings.append(state.detach())
            
        non_depression_embeddings.extend(scenario_embeddings)
    
    # Convert to numpy arrays
    depression_embeddings = torch.stack(depression_embeddings).squeeze(1).numpy()
    non_depression_embeddings = torch.stack(non_depression_embeddings).squeeze(1).numpy()
    
    # Combine all embeddings for UMAP
    all_embeddings = np.vstack([depression_embeddings, non_depression_embeddings])
    labels = np.array(
        ["Depression"] * len(depression_embeddings) + 
        ["Non-Depression"] * len(non_depression_embeddings)
    )
    
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=15,  # Local neighborhood size
        min_dist=0.1,    # Minimum distance between points
        metric='cosine'  # Use cosine similarity for emotional embeddings
    )
    embeddings_2d = reducer.fit_transform(all_embeddings)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(
        all_embeddings,
        labels == "Depression"
    )
    
    # Plot results with improved styling
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create color palette
    colors = sns.color_palette("husl", 2)
    
    # Plot depression points
    plt.scatter(
        embeddings_2d[:len(depression_embeddings), 0],
        embeddings_2d[:len(depression_embeddings), 1],
        c=[colors[0]],
        label="Depression",
        alpha=0.7,
        s=100
    )
    
    # Plot non-depression points
    plt.scatter(
        embeddings_2d[len(depression_embeddings):, 0],
        embeddings_2d[len(depression_embeddings):, 1],
        c=[colors[1]],
        label="Non-Depression",
        alpha=0.7,
        s=100
    )
    
    plt.title(
        "Emotional State Embeddings (UMAP)\n" +
        f"Silhouette Score: {silhouette_avg:.3f}",
        fontsize=14,
        pad=20
    )
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # Add legend with larger font
    plt.legend(title="Emotional State", title_fontsize=12, fontsize=10)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(
        output_dir / "depression_analysis_umap.png",
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
    
    return silhouette_avg

if __name__ == "__main__":
    silhouette_score = analyze_emotional_trajectories()
    print(f"\nAnalysis complete! Silhouette Score: {silhouette_score:.3f}")
    print("Visualization saved as 'depression_analysis_umap.png'")
