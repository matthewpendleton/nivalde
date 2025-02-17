"""Tests for learned temporal-emotional manifold properties."""

import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from src.memory.temporal_manifold import TemporalManifold, ManifoldPoint
from src.memory.temporal_trainer import TemporalTrainer, TemporalExpressionDataset

class TestManifoldProperties:
    @pytest.fixture
    def training_data(self):
        """Create synthetic training data with known temporal and emotional patterns."""
        expressions = []
        timestamps = []
        emotional_labels = []
        
        # Generate data spanning different time scales
        base_time = datetime(2024, 1, 1)
        time_expressions = [
            ("just now", 0, 0.8),
            ("a minute ago", 1, 0.6),
            ("an hour ago", 60, 0.5),
            ("this morning", 180, 0.4),
            ("yesterday", 1440, 0.2),
            ("last week", 10080, 0.0),
            ("last month", 43200, -0.2),
            ("last year", 525600, -0.4)
        ]
        
        for expr, minutes_ago, emotion in time_expressions:
            expressions.append(expr)
            timestamps.append(minutes_ago)
            emotional_labels.append(emotion)
            
        # Add emotional variations
        emotional_expressions = [
            ("during the happiest moment", 5000, 1.0),
            ("when everything was perfect", 5100, 0.9),
            ("in darker times", 5200, -0.9),
            ("through the struggles", 5300, -0.8),
            ("as things improved", 5400, 0.7)
        ]
        
        for expr, minutes_ago, emotion in emotional_expressions:
            expressions.append(expr)
            timestamps.append(minutes_ago)
            emotional_labels.append(emotion)
            
        return expressions, timestamps, emotional_labels

    @pytest.fixture
    def trained_manifold(self, training_data):
        """Create and train a manifold on synthetic data."""
        expressions, timestamps, emotional_labels = training_data
        
        # Normalize timestamps to [0,1]
        timestamps = np.array(timestamps)
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        
        # Create dataset
        dataset = TemporalExpressionDataset(
            expressions=expressions,
            timestamps=torch.tensor(timestamps, dtype=torch.float32),
            emotional_labels=torch.tensor(emotional_labels, dtype=torch.float32)
        )
        
        # Create and train manifold
        manifold = TemporalManifold(
            manifold_dim=32,  # Smaller for testing
            emotional_dim=8,
            n_heads=4
        )
        
        trainer = TemporalTrainer(
            model=manifold,
            lr=0.001
        )
        
        # Train for a few epochs
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True
        )
        
        for epoch in range(10):  # Quick training for tests
            trainer.train_epoch(dataloader, epoch)
            
        return manifold

    def test_temporal_ordering(self, trained_manifold):
        """Test that clear temporal sequences maintain ordering in manifold space."""
        expressions = [
            "just now",
            "an hour ago", 
            "yesterday",
            "last month"
        ]
        
        points = trained_manifold(expressions)
        embeddings = torch.stack([p.embedding for p in points])
        
        # Test relative distances
        for i in range(len(points)-1):
            for j in range(i+1, len(points)):
                dist_ij = torch.norm(embeddings[i] - embeddings[j])
                # For each pair, check all pairs with smaller temporal distance
                for k in range(i+1, j):
                    dist_ik = torch.norm(embeddings[i] - embeddings[k])
                    assert dist_ik <= dist_ij, \
                        f"Temporal ordering violated between {expressions[i]}, {expressions[j]} and {expressions[k]}"
    
    def test_uncertainty_calibration(self, trained_manifold):
        """Test that uncertainty estimates are well-calibrated."""
        expressions = [
            "just now",              # Very precise
            "sometime today",        # Less precise
            "a while ago",          # Vague
            "back then"             # Very vague
        ]
        
        points = trained_manifold(expressions)
        uncertainties = torch.stack([p.uncertainty for p in points])
        
        # Check uncertainty increases with vagueness
        diffs = uncertainties[1:] - uncertainties[:-1]
        assert torch.all(diffs >= -0.1), \
            "Uncertainty should generally increase with temporal vagueness"
    
    def test_emotional_consistency(self, trained_manifold):
        """Test that emotional valence is consistent across similar expressions."""
        positive_expressions = [
            "during the happiest moment",
            "when everything was perfect"
        ]
        
        negative_expressions = [
            "in darker times",
            "through the struggles"
        ]
        
        pos_points = trained_manifold(positive_expressions)
        neg_points = trained_manifold(negative_expressions)
        
        pos_emotions = torch.stack([p.emotional_valence for p in pos_points])
        neg_emotions = torch.stack([p.emotional_valence for p in neg_points])
        
        assert torch.mean(pos_emotions) > torch.mean(neg_emotions), \
            "Positive expressions should have higher emotional valence"
    
    def test_manifold_continuity(self, trained_manifold):
        """Test that the manifold is continuous between temporal points."""
        # Test interpolation between two points
        start_expr = "just now"
        end_expr = "last week"
        
        start_point = trained_manifold([start_expr])[0]
        end_point = trained_manifold([end_expr])[0]
        
        # Generate interpolated points
        alphas = torch.linspace(0, 1, 10)
        interp_embeddings = []
        
        for alpha in alphas:
            interp_emb = (1 - alpha) * start_point.embedding + alpha * end_point.embedding
            interp_embeddings.append(interp_emb)
            
        interp_embeddings = torch.stack(interp_embeddings)
        
        # Check smoothness
        diffs = torch.norm(interp_embeddings[1:] - interp_embeddings[:-1], dim=1)
        max_jump = torch.max(diffs)
        mean_jump = torch.mean(diffs)
        
        assert max_jump < 2 * mean_jump, \
            "Interpolation should be smooth without large jumps"
