"""Test suite for analyzing personality disorder patterns and trajectories."""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics.pairwise import cosine_similarity

from src.emotional_embedding.ees import EmotionalEmbeddingSpace
from src.memory.memory_system import TransformerMemorySystem
from tests.test_data.personality_disorder_scenarios import (
    TEST_VIGNETTE,
    PERSONALITY_SCENARIOS,
    PERSONALITY_MARKERS
)

def process_personality_scenario(
    ees_model: EmotionalEmbeddingSpace,
    memory_system: TransformerMemorySystem,
    dialogue: list,
    test_vignette: list
) -> tuple:
    """Process a personality scenario through EES.
    
    Args:
        ees_model: Emotional embedding model
        memory_system: Memory system
        dialogue: List of dialogue entries
        test_vignette: List of test situation entries
        
    Returns:
        Tuple of (personality embeddings, vignette embeddings)
    """
    personality_embeddings = []
    memory_context = torch.zeros(1, 768)
    
    # Process personality dialogue
    for utterance in dialogue:
        current_embedding = torch.randn(1, 768)
        state = ees_model(
            current_state=current_embedding,
            bert_context=current_embedding,
            memory_context=memory_context
        )
        memory_context = memory_system(
            current_input=current_embedding,
            previous_memory=memory_context
        )
        personality_embeddings.append(state.detach())
    
    # Process test vignette
    vignette_embeddings = []
    for utterance in test_vignette:
        current_embedding = torch.randn(1, 768)
        state = ees_model(
            current_state=current_embedding,
            bert_context=current_embedding,
            memory_context=memory_context
        )
        memory_context = memory_system(
            current_input=current_embedding,
            previous_memory=memory_context
        )
        vignette_embeddings.append(state.detach())
    
    return (
        torch.stack(personality_embeddings).squeeze(1),
        torch.stack(vignette_embeddings).squeeze(1)
    )

def analyze_personality_patterns():
    """Analyze personality disorder patterns and their influence on interpretation."""
    # Initialize models
    ees = EmotionalEmbeddingSpace(dim=768)
    memory_system = TransformerMemorySystem(embedding_dim=768)
    
    # Process scenarios
    personality_embeddings = {}
    vignette_embeddings = {}
    
    for scenario_id, scenario in PERSONALITY_SCENARIOS.items():
        personality_emb, vignette_emb = process_personality_scenario(
            ees,
            memory_system,
            scenario["dialogue"],
            TEST_VIGNETTE
        )
        personality_embeddings[scenario_id] = personality_emb
        vignette_embeddings[scenario_id] = vignette_emb
    
    # Combine all embeddings for visualization
    all_embeddings = []
    all_labels = []
    
    for scenario_id, emb in personality_embeddings.items():
        all_embeddings.append(emb)
        all_labels.extend([scenario_id] * len(emb))
    
    all_embeddings = torch.cat(all_embeddings).numpy()
    
    # UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine'
    )
    embeddings_2d = reducer.fit_transform(all_embeddings)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # Create color palette
    unique_disorders = set(s.split("_")[0] for s in PERSONALITY_SCENARIOS.keys())
    colors = sns.color_palette("husl", len(unique_disorders))
    color_map = {d: c for d, c in zip(unique_disorders, colors)}
    
    # Plot each personality pattern
    start_idx = 0
    for scenario_id, emb in personality_embeddings.items():
        disorder_type = scenario_id.split("_")[0]
        end_idx = start_idx + len(emb)
        
        plt.scatter(
            embeddings_2d[start_idx:end_idx, 0],
            embeddings_2d[start_idx:end_idx, 1],
            c=[color_map[disorder_type]],
            label=scenario_id.replace("_", " ").title(),
            alpha=0.7,
            s=100
        )
        
        # Add trajectory arrows
        for i in range(start_idx, end_idx-1):
            plt.arrow(
                embeddings_2d[i, 0],
                embeddings_2d[i, 1],
                embeddings_2d[i+1, 0] - embeddings_2d[i, 0],
                embeddings_2d[i+1, 1] - embeddings_2d[i, 1],
                head_width=0.1,
                head_length=0.1,
                fc=color_map[disorder_type],
                ec=color_map[disorder_type],
                alpha=0.3
            )
        
        start_idx = end_idx
    
    plt.title(
        "Personality Patterns in Emotional Space\n" +
        "(Trajectories show emotional evolution through dialogue)",
        fontsize=14,
        pad=20
    )
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # Add legend
    plt.legend(
        title="Personality Pattern",
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
        output_dir / "personality_patterns.png",
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
    
    # Analyze vignette interpretations
    print("\nAnalysis of Test Vignette Interpretations:")
    print("----------------------------------------")
    
    # Convert final vignette states to numpy
    vignette_final_states = {
        k: v[-1].numpy() for k, v in vignette_embeddings.items()
    }
    
    # Compute similarities between different interpretations
    for id1 in vignette_final_states.keys():
        for id2 in vignette_final_states.keys():
            if id1 < id2:
                sim = cosine_similarity(
                    vignette_final_states[id1].reshape(1, -1),
                    vignette_final_states[id2].reshape(1, -1)
                )[0, 0]
                
                print(f"{id1.replace('_', ' ').title()} vs "
                      f"{id2.replace('_', ' ').title()}: "
                      f"{sim:.3f}")
    
    # Analyze age-related patterns
    print("\nAge-Related Pattern Analysis:")
    print("----------------------------")
    
    for disorder in ["borderline", "narcissistic", "avoidant", "obsessive_compulsive"]:
        young_id = f"{disorder}_young"
        older_id = f"{disorder}_middle" if disorder in ["borderline", "avoidant"] else f"{disorder}_older"
        
        if young_id in vignette_final_states and older_id in vignette_final_states:
            sim = cosine_similarity(
                vignette_final_states[young_id].reshape(1, -1),
                vignette_final_states[older_id].reshape(1, -1)
            )[0, 0]
            
            print(f"{disorder.title()} (Young vs Older): {sim:.3f}")

if __name__ == "__main__":
    analyze_personality_patterns()
