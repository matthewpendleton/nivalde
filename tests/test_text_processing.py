import sys
import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import umap
from collections import defaultdict

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.input.multimodal_processor import MultimodalProcessor

def visualize_embeddings(embeddings_list, labels, categories, title):
    """Visualize embeddings using UMAP with color-coded categories"""
    # Convert embeddings to numpy array
    embeddings_array = torch.stack(embeddings_list).numpy()
    
    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    embeddings_2d = reducer.fit_transform(embeddings_array)
    
    # Create color map for categories
    unique_categories = list(set(categories))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    category_colors = dict(zip(unique_categories, colors))
    
    # Plot
    plt.figure(figsize=(15, 12))
    
    # Plot points by category
    for category in unique_categories:
        # Get indices for this category
        indices = [i for i, c in enumerate(categories) if c == category]
        points = embeddings_2d[indices]
        category_labels = [labels[i] for i in indices]
        
        # Plot points
        plt.scatter(points[:, 0], points[:, 1], 
                   color=category_colors[category], 
                   label=category, 
                   alpha=0.7)
        
        # Add labels with smaller font and slight offset
        for i, label in enumerate(category_labels):
            plt.annotate(label, 
                        (points[i, 0], points[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=6, alpha=0.8)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('UMAP dimension 1')
    plt.ylabel('UMAP dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Save plot
    output_dir = Path(project_root) / 'tests' / 'output'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'{title.lower().replace(" ", "_")}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def calculate_emotion_metrics(embeddings_list, categories):
    """Calculate metrics about emotional embedding patterns"""
    embeddings_array = torch.stack(embeddings_list).numpy()
    
    # Group embeddings by category
    category_embeddings = {}
    for i, category in enumerate(categories):
        if category not in category_embeddings:
            category_embeddings[category] = []
        category_embeddings[category].append(embeddings_array[i])
    
    # Calculate metrics
    metrics = {
        "intra_cluster_distances": {},
        "inter_cluster_distances": {},
        "cluster_cohesion": {},
        "cluster_separation": {}
    }
    
    # Calculate intra-cluster distances (cohesion)
    for category, embeddings in category_embeddings.items():
        if len(embeddings) > 1:
            distances = []
            embeddings = np.array(embeddings)
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    distances.append(dist)
            metrics["intra_cluster_distances"][category] = np.mean(distances)
            metrics["cluster_cohesion"][category] = 1 / (1 + np.mean(distances))
    
    # Calculate inter-cluster distances (separation)
    categories = list(category_embeddings.keys())
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            cat1, cat2 = categories[i], categories[j]
            distances = []
            for emb1 in category_embeddings[cat1]:
                for emb2 in category_embeddings[cat2]:
                    dist = np.linalg.norm(emb1 - emb2)
                    distances.append(dist)
            key = f"{cat1} vs {cat2}"
            metrics["inter_cluster_distances"][key] = np.mean(distances)
            metrics["cluster_separation"][key] = np.mean(distances)
    
    return metrics

def print_emotion_analysis(metrics):
    """Print analysis of emotional embedding patterns"""
    print("\nEmotion Embedding Analysis:")
    print("\n1. Cluster Cohesion (lower is better, indicates tighter clustering):")
    for category, distance in metrics["intra_cluster_distances"].items():
        print(f"   {category}: {distance:.4f}")
    
    print("\n2. Cluster Separation (higher is better, indicates distinct emotions):")
    for pair, distance in metrics["inter_cluster_distances"].items():
        print(f"   {pair}: {distance:.4f}")
    
    print("\n3. Category Distinctiveness (cohesion / separation ratio):")
    for category in metrics["cluster_cohesion"]:
        cohesion = metrics["cluster_cohesion"][category]
        separations = []
        for pair, sep in metrics["cluster_separation"].items():
            if category in pair:
                separations.append(sep)
        if separations:
            avg_separation = np.mean(separations)
            distinctiveness = cohesion * avg_separation
            print(f"   {category}: {distinctiveness:.4f}")

def main():
    # Initialize processor
    processor = MultimodalProcessor()
    
    # Test sentences grouped by emotional categories with sophisticated variations
    emotional_groups = {
        "Complex Emotional States": [
            "I'm experiencing a mix of anticipation and anxiety about tomorrow's presentation.",
            "While generally optimistic, I can't shake this underlying sense of unease.",
            "My excitement is tinged with a hint of melancholy, knowing this chapter is ending.",
            "I feel both grateful for the opportunity and apprehensive about the responsibility.",
            "There's a bittersweet satisfaction in completing this journey.",
        ],
        "Temporal Emotional Shifts": [
            "My initial frustration has gradually transformed into acceptance.",
            "What started as mild concern has evolved into deep-seated worry.",
            "The joy I felt yesterday has been replaced by a profound sense of loss.",
            "My anxiety peaks in the morning but settles into calm by evening.",
            "The disappointment fades a little more with each passing day.",
        ],
        "Emotional Intensity Gradients": [
            "I'm slightly annoyed by the delay.",
            "I'm getting increasingly frustrated with these setbacks.",
            "I'm thoroughly exasperated by this entire situation.",
            "I'm completely overwhelmed with anger at this point.",
            "I'm absolutely livid about how this was handled.",
        ],
        "Conditional Emotions": [
            "I would feel more confident if I had more time to prepare.",
            "My anxiety lessens when I focus on breathing deeply.",
            "The sadness comes in waves, especially during quiet moments.",
            "My joy multiplies whenever I share it with others.",
            "The stress intensifies as the deadline approaches.",
        ],
        "Emotional Cause-Effect": [
            "Their unexpected praise filled me with a surge of pride and motivation.",
            "The constant criticism has eroded my once-steady confidence.",
            "Each small success builds my resilience against future setbacks.",
            "The uncertainty of the situation feeds my growing anxiety.",
            "Your support transforms my self-doubt into determination.",
        ],
        "Comparative Emotions": [
            "Unlike yesterday's despair, today I feel cautiously hopeful.",
            "My current contentment feels deeper than my previous happiness.",
            "This fear is different from my usual anxiety - it's more primal.",
            "Today's disappointment stings more than all previous setbacks combined.",
            "My joy now feels more genuine than the superficial excitement before.",
        ],
        "Emotional Self-Awareness": [
            "I recognize that my anger might be masking deeper hurt.",
            "Perhaps my excessive enthusiasm compensates for underlying insecurity.",
            "I'm aware that my perfectionism stems from fear of failure.",
            "My outward calm belies significant internal turmoil.",
            "This optimism feels like a conscious choice rather than natural emotion.",
        ],
        "Social-Context Emotions": [
            "In professional settings, I suppress my frustration behind a polite smile.",
            "Among friends, my happiness flows more freely and authentically.",
            "During family gatherings, my anxiety tends to peak then gradually subside.",
            "My confidence wavers when presenting to senior management.",
            "In competitive situations, my determination overshadows my fear.",
        ]
    }
    
    # Process each sentence and collect embeddings
    text_embeddings = []
    contextual_embeddings = []
    labels = []
    categories = []
    
    print("\nProcessing test sentences...")
    for category, sentences in emotional_groups.items():
        for sentence in sentences:
            print(f"\nInput ({category}): {sentence}")
            
            # Process input
            result = processor.process_input(text=sentence)
            
            # Get embeddings
            text_emb = result.text_embedding.squeeze()
            ctx_emb = result.contextual_embedding.squeeze()
            
            # Print shapes and sample values
            print(f"Text embedding shape: {text_emb.shape}")
            print(f"Contextual embedding shape: {ctx_emb.shape}")
            
            # Store embeddings and metadata
            text_embeddings.append(text_emb)
            contextual_embeddings.append(ctx_emb)
            
            # Extract the key emotion word (usually after "feeling" or "feel")
            words = sentence.split()
            if "feeling" in words:
                emotion = words[words.index("feeling") + 1]
            elif "feel" in words:
                emotion = words[words.index("feel") + 1]
            else:
                emotion = words[2]  # fallback to third word
            
            labels.append(emotion)
            categories.append(category)
    
    # Visualize embeddings
    print("\nGenerating visualizations...")
    visualize_embeddings(text_embeddings, labels, categories, 
                        "Raw Text Embeddings by Emotion Category")
    visualize_embeddings(contextual_embeddings, labels, categories,
                        "Contextual Embeddings by Emotion Category")
    
    # Calculate and print emotion metrics
    metrics = calculate_emotion_metrics(text_embeddings, categories)
    print_emotion_analysis(metrics)
    
    print("\nTest complete! Check the tests/output directory for visualization plots.")

if __name__ == "__main__":
    main()
