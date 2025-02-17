"""Training script for temporal-emotional manifold learning."""

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import pandas as pd
from src.memory.temporal_manifold import TemporalManifold
from src.memory.temporal_trainer import TemporalTrainer, TemporalExpressionDataset, collate_temporal_batch
from src.memory.manifold_viz import ManifoldVisualizer

def generate_training_data() -> Tuple[List[str], List[float], List[float]]:
    """Generate rich training data for temporal expressions."""
    expressions = []
    timestamps = []
    emotional_labels = []
    
    # Base time for relative expressions
    now = datetime.now()
    
    # 1. Precise time expressions
    precise_times = [
        ("just now", 0, 0.0),
        ("a minute ago", 1, 0.0),
        ("five minutes ago", 5, 0.0),
        ("half an hour ago", 30, 0.0),
        ("an hour ago", 60, 0.0),
        ("this morning", 180, 0.2),
        ("yesterday evening", 1200, -0.1),
        ("yesterday morning", 1440, 0.0),
        ("two days ago", 2880, 0.0),
        ("last week", 10080, 0.0),
        ("a month ago", 43200, 0.0),
    ]
    
    for expr, minutes_ago, emotion in precise_times:
        expressions.append(expr)
        timestamps.append(minutes_ago)
        emotional_labels.append(emotion)
    
    # 2. Emotional life events
    emotional_events = [
        ("when I first fell in love", 525600, 0.9),
        ("during the darkest time", 262800, -0.9),
        ("as things started improving", 131400, 0.5),
        ("when hope returned", 65700, 0.8),
        ("before I understood", 32850, -0.3),
        ("after I learned to trust", 16425, 0.7),
        ("while still healing", 8212, 0.4),
        ("as wisdom grew", 4106, 0.6),
    ]
    
    for expr, minutes_ago, emotion in emotional_events:
        expressions.append(expr)
        timestamps.append(minutes_ago)
        emotional_labels.append(emotion)
    
    # 3. Vague temporal expressions
    vague_times = [
        ("a while ago", 10000, 0.0),
        ("some time back", 20000, 0.0),
        ("in the past", 50000, -0.1),
        ("recently", 5000, 0.1),
        ("not too long ago", 7500, 0.0),
        ("ages ago", 100000, -0.2),
    ]
    
    for expr, minutes_ago, emotion in vague_times:
        expressions.append(expr)
        timestamps.append(minutes_ago)
        emotional_labels.append(emotion)
    
    # 4. Life stage references
    life_stages = [
        ("in my childhood", 5256000, 0.3),  # ~10 years ago
        ("during my teens", 3153600, -0.1),  # ~6 years ago
        ("when I was younger", 2102400, 0.2),  # ~4 years ago
        ("in recent years", 525600, 0.1),  # ~1 year ago
        ("these days", 1440, 0.4),  # Now-ish
    ]
    
    for expr, minutes_ago, emotion in life_stages:
        expressions.append(expr)
        timestamps.append(minutes_ago)
        emotional_labels.append(emotion)
    
    # 5. Emotional intensity variations
    base_events = [
        "when I felt",
        "as I experienced",
        "during the time of",
        "in moments of",
    ]
    
    emotions = [
        ("joy", 0.9),
        ("sadness", -0.8),
        ("peace", 0.7),
        ("anxiety", -0.7),
        ("love", 0.9),
        ("fear", -0.8),
        ("hope", 0.8),
        ("doubt", -0.6),
    ]
    
    for base in base_events:
        for emotion, valence in emotions:
            expr = f"{base} {emotion}"
            # Random time between 1 day and 2 years ago
            minutes_ago = np.random.randint(1440, 1051200)
            expressions.append(expr)
            timestamps.append(minutes_ago)
            emotional_labels.append(valence)
    
    # 6. Sequential events
    sequence = [
        ("before it all began", 100000, -0.2),
        ("when it started", 90000, -0.4),
        ("as things unfolded", 80000, -0.1),
        ("during the change", 70000, 0.2),
        ("after the transformation", 60000, 0.6),
        ("now that it's done", 50000, 0.8),
    ]
    
    for expr, minutes_ago, emotion in sequence:
        expressions.append(expr)
        timestamps.append(minutes_ago)
        emotional_labels.append(emotion)
    
    return expressions, timestamps, emotional_labels

def train_manifold():
    """Train the temporal-emotional manifold."""
    # Generate training data
    expressions, timestamps, emotional_labels = generate_training_data()
    
    # Normalize timestamps to [0,1]
    timestamps = np.array(timestamps)
    timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    
    # Create dataset
    dataset = TemporalExpressionDataset(
        expressions=expressions,
        timestamps=torch.tensor(timestamps, dtype=torch.float32),
        emotional_labels=torch.tensor(emotional_labels, dtype=torch.float32)
    )
    
    # Create model and trainer
    manifold = TemporalManifold(
        manifold_dim=64,
        emotional_dim=16,
        n_heads=8
    )
    
    trainer = TemporalTrainer(
        model=manifold,
        lr=0.001
    )
    
    # Training loop
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_temporal_batch
    )
    
    print("Starting training...")
    for epoch in range(100):
        metrics = trainer.train_epoch(dataloader, epoch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {metrics}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    viz = ManifoldVisualizer(manifold)
    
    # Test expressions
    test_expressions = [
        "just now",
        "yesterday",
        "last week",
        "during my happiest moment",
        "in the darkest times",
        "as hope returned",
        "while healing",
        "before understanding",
        "after growth",
        "nowadays"
    ]
    
    viz.create_interactive_dashboard(
        test_expressions,
        save_path="temporal_manifold_results.html"
    )
    
    print("\nTesting learned relationships...")
    # Test some relationships
    def get_temporal_distance(expr1: str, expr2: str) -> float:
        points = manifold([expr1, expr2])
        emb1, emb2 = points[0].embedding, points[1].embedding
        return torch.norm(emb1 - emb2).item()
    
    def get_emotional_valence(expr: str) -> float:
        point = manifold([expr])[0]
        return point.emotional_valence.item()
    
    # Test temporal ordering
    print("\nTemporal distances:")
    time_pairs = [
        ("just now", "yesterday"),
        ("yesterday", "last week"),
        ("last week", "last month"),
    ]
    
    for expr1, expr2 in time_pairs:
        dist = get_temporal_distance(expr1, expr2)
        print(f"{expr1} -> {expr2}: {dist:.3f}")
    
    # Test emotional understanding
    print("\nEmotional valences:")
    emotion_exprs = [
        "during my happiest moment",
        "in the darkest times",
        "as hope returned",
        "while healing"
    ]
    
    for expr in emotion_exprs:
        valence = get_emotional_valence(expr)
        print(f"{expr}: {valence:.3f}")
    
    return manifold

if __name__ == "__main__":
    trained_manifold = train_manifold()
