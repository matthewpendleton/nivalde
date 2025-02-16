"""Test suite for analyzing how memory influences emotional embeddings."""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics.pairwise import cosine_similarity

from src.emotional_embedding.ees import EmotionalEmbeddingSpace
from src.memory.memory_system import TransformerMemorySystem
from tests.test_data.memory_influence_scenarios import (
    TEST_VIGNETTE,
    PATIENT_HISTORIES,
    EMOTIONAL_MARKERS
)

def process_patient_scenario(
    ees_model: EmotionalEmbeddingSpace,
    memory_system: TransformerMemorySystem,
    history_dialogue: list,
    test_vignette: list
) -> tuple:
    """Process a patient's history and test vignette through EES.
    
    Args:
        ees_model: Emotional embedding model
        memory_system: Memory system
        history_dialogue: List of historical dialogue entries
        test_vignette: List of current situation dialogue entries
        
    Returns:
        Tuple of (history embeddings, vignette embeddings)
    """
    history_embeddings = []
    memory_context = torch.zeros(1, 768)  # Initial memory context
    
    # Process history
    for utterance in history_dialogue:
        # Simulate BERT embeddings
        current_embedding = torch.randn(1, 768)
        
        # Get emotional state
        state = ees_model(
            current_state=current_embedding,
            bert_context=current_embedding,
            memory_context=memory_context
        )
        
        # Update memory
        memory_context = memory_system(
            current_input=current_embedding,
            previous_memory=memory_context
        )
        
        history_embeddings.append(state.detach())
    
    # Process test vignette with loaded memory
    vignette_embeddings = []
    for utterance in test_vignette:
        # Simulate BERT embeddings
        current_embedding = torch.randn(1, 768)
        
        # Get emotional state
        state = ees_model(
            current_state=current_embedding,
            bert_context=current_embedding,
            memory_context=memory_context
        )
        
        # Update memory
        memory_context = memory_system(
            current_input=current_embedding,
            previous_memory=memory_context
        )
        
        vignette_embeddings.append(state.detach())
    
    return (
        torch.stack(history_embeddings).squeeze(1),
        torch.stack(vignette_embeddings).squeeze(1)
    )

def analyze_memory_influence():
    """Analyze how different patient histories influence interpretation."""
    # Initialize models
    ees = EmotionalEmbeddingSpace(dim=768)
    memory_system = TransformerMemorySystem(embedding_dim=768)
    
    # Process each patient history
    history_embeddings = {}
    vignette_embeddings = {}
    
    for history_type, scenario in PATIENT_HISTORIES.items():
        history_emb, vignette_emb = process_patient_scenario(
            ees,
            memory_system,
            scenario["dialogue"],
            TEST_VIGNETTE
        )
        
        history_embeddings[history_type] = history_emb
        vignette_embeddings[history_type] = vignette_emb
    
    # Combine all embeddings for visualization
    all_vignette_embeddings = torch.cat(list(vignette_embeddings.values()))
    
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine'
    )
    embeddings_2d = reducer.fit_transform(all_vignette_embeddings.numpy())
    
    # Plot results
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # Create color palette
    colors = sns.color_palette("husl", len(PATIENT_HISTORIES))
    
    # Plot each patient's vignette interpretations
    start_idx = 0
    for i, (history_type, emb) in enumerate(vignette_embeddings.items()):
        end_idx = start_idx + len(TEST_VIGNETTE)
        
        plt.scatter(
            embeddings_2d[start_idx:end_idx, 0],
            embeddings_2d[start_idx:end_idx, 1],
            c=[colors[i]],
            label=history_type.replace("_", " ").title(),
            alpha=0.7,
            s=100
        )
        
        # Add trajectory arrows
        for j in range(start_idx, end_idx-1):
            plt.arrow(
                embeddings_2d[j, 0],
                embeddings_2d[j, 1],
                embeddings_2d[j+1, 0] - embeddings_2d[j, 0],
                embeddings_2d[j+1, 1] - embeddings_2d[j, 1],
                head_width=0.1,
                head_length=0.1,
                fc=colors[i],
                ec=colors[i],
                alpha=0.5
            )
        
        start_idx = end_idx
    
    plt.title(
        "Influence of Patient History on Current Situation Interpretation\n" +
        "(Trajectories show progression through test vignette)",
        fontsize=14,
        pad=20
    )
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # Add legend with larger font
    plt.legend(
        title="Patient History",
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(
        output_dir / "memory_influence.png",
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
    
    # Compute similarity analysis
    print("\nSimilarity Analysis of Vignette Interpretations:")
    print("-----------------------------------------------")
    
    # Convert embeddings to numpy for similarity computation
    vignette_final_states = {
        k: v[-1].numpy() for k, v in vignette_embeddings.items()
    }
    
    # Compute pairwise similarities
    for h1 in vignette_final_states.keys():
        for h2 in vignette_final_states.keys():
            if h1 < h2:  # Only compute upper triangle
                sim = cosine_similarity(
                    vignette_final_states[h1].reshape(1, -1),
                    vignette_final_states[h2].reshape(1, -1)
                )[0, 0]
                
                print(f"{h1.replace('_', ' ').title()} vs "
                      f"{h2.replace('_', ' ').title()}: "
                      f"{sim:.3f}")

if __name__ == "__main__":
    analyze_memory_influence()
