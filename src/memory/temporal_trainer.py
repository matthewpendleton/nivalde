"""
Training system for temporal-emotional manifold learning.

This module handles:
1. Data loading from various sources (therapy transcripts, social media, etc.)
2. Training loop with curriculum learning
3. Manifold evaluation and visualization
4. Active learning for uncertainty calibration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime
import pandas as pd
from collections import defaultdict

from .temporal_manifold import TemporalManifold, ManifoldPoint, TemporalLoss

class TemporalExpressionDataset(Dataset):
    """Dataset for temporal expressions with emotional context."""
    
    def __init__(
        self,
        expressions: List[str],
        timestamps: List[float],
        emotional_labels: Optional[List[float]] = None,
        context_size: int = 5
    ):
        self.expressions = expressions
        self.timestamps = torch.tensor(timestamps, dtype=torch.float32)
        self.emotional_labels = torch.tensor(emotional_labels, dtype=torch.float32) if emotional_labels is not None else None
        self.context_size = context_size
        
        # Normalize timestamps to [0, 1]
        self.norm_timestamps = (self.timestamps - self.timestamps.min()) / (
            self.timestamps.max() - self.timestamps.min()
        )
        
    def __len__(self):
        return len(self.expressions)
        
    def __getitem__(self, idx):
        # Get context window with padding
        start_idx = max(0, idx - self.context_size)
        context_exprs = self.expressions[start_idx:idx]
        context_times = self.norm_timestamps[start_idx:idx].tolist()
        
        # Pad if needed
        pad_length = self.context_size - len(context_exprs)
        if pad_length > 0:
            context_exprs = ['<pad>'] * pad_length + context_exprs
            context_times = [0.0] * pad_length + context_times
            
        item = {
            'expression': self.expressions[idx],
            'timestamp': self.norm_timestamps[idx],
            'context_expressions': context_exprs,
            'context_timestamps': torch.tensor(context_times, dtype=torch.float32)
        }
        
        if self.emotional_labels is not None:
            item['emotional_label'] = self.emotional_labels[idx]
            
        return item

def collate_temporal_batch(batch):
    """Custom collate function for temporal batches."""
    return {
        'expression': [item['expression'] for item in batch],
        'timestamp': torch.stack([item['timestamp'] for item in batch]),
        'context_expressions': [item['context_expressions'] for item in batch],
        'context_timestamps': torch.stack([item['context_timestamps'] for item in batch]),
        'emotional_label': torch.stack([item['emotional_label'] for item in batch]) if 'emotional_label' in batch[0] else None
    }

class TemporalTrainer:
    """Trainer for temporal-emotional manifold."""
    
    def __init__(
        self,
        model,
        lr: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.criterion = TemporalLoss()
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        metrics = defaultdict(float)
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Process expressions
            expressions = batch['expression']
            timestamps = batch['timestamp'].to(self.device)
            context_expressions = batch['context_expressions']
            context_timestamps = batch['context_timestamps'].to(self.device)
            emotional_labels = batch.get('emotional_label')
            if emotional_labels is not None:
                emotional_labels = emotional_labels.to(self.device)
                
            # Forward pass
            points = self.model(
                expressions,
                context_expressions=context_expressions,
                context_timestamps=context_timestamps
            )
            
            # Compute loss
            loss = self.criterion(points, timestamps, emotional_labels)
            total_loss += loss.item()
            
            # Compute metrics
            with torch.no_grad():
                # Get embeddings
                embeddings = torch.stack([p.embedding for p in points])
                
                # Compute pairwise distances
                pred_dists = torch.cdist(embeddings, embeddings)
                true_dists = torch.cdist(timestamps.unsqueeze(1), timestamps.unsqueeze(1))
                
                # Flatten the distance matrices for correlation
                pred_flat = pred_dists.view(-1)
                true_flat = true_dists.view(-1)
                
                # Remove diagonal elements
                mask = ~torch.eye(pred_dists.shape[0], dtype=bool).view(-1)
                pred_flat = pred_flat[mask]
                true_flat = true_flat[mask]
                
                # Calculate correlation
                order_acc = torch.corrcoef(
                    torch.stack([pred_flat, true_flat])
                )[0, 1].item()
                
                metrics['order_acc'] += order_acc
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
        # Average metrics
        metrics = {k: v / len(dataloader) for k, v in metrics.items()}
        metrics['loss'] = total_loss / len(dataloader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {metrics['loss']:.4f}, Order Acc = {metrics['order_acc']:.4f}")
        
        return metrics
        
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        metrics = defaultdict(float)
        
        with torch.no_grad():
            for batch in dataloader:
                expressions = batch['expression']
                timestamps = batch['timestamp'].to(self.device)
                context_expressions = batch['context_expressions']
                context_timestamps = batch['context_timestamps'].to(self.device)
                emotional_labels = batch.get('emotional_label')
                if emotional_labels is not None:
                    emotional_labels = emotional_labels.to(self.device)
                    
                # Forward pass
                points = self.model(
                    expressions,
                    context_expressions=context_expressions,
                    context_timestamps=context_timestamps
                )
                
                # Get embeddings
                embeddings = torch.stack([p.embedding for p in points])
                
                # Compute pairwise distances
                pred_dists = torch.cdist(embeddings, embeddings)
                true_dists = torch.cdist(timestamps.unsqueeze(1), timestamps.unsqueeze(1))
                
                # Flatten the distance matrices for correlation
                pred_flat = pred_dists.view(-1)
                true_flat = true_dists.view(-1)
                
                # Remove diagonal elements
                mask = ~torch.eye(pred_dists.shape[0], dtype=bool).view(-1)
                pred_flat = pred_flat[mask]
                true_flat = true_flat[mask]
                
                # Calculate correlation
                order_acc = torch.corrcoef(
                    torch.stack([pred_flat, true_flat])
                )[0, 1].item()
                
                metrics['val_order_acc'] += order_acc
                
        # Average metrics
        metrics = {k: v / len(dataloader) for k, v in metrics.items()}
        
        return metrics
        
    def visualize_manifold(
        self,
        expressions: List[str],
        save_path: Optional[str] = None
    ):
        """Visualize points on the temporal-emotional manifold."""
        self.model.eval()
        
        with torch.no_grad():
            points = self.model(expressions)
            
            # Get embeddings and reduce to 2D
            embeddings = torch.stack([p.embedding for p in points])
            uncertainties = torch.stack([p.uncertainty for p in points])
            emotional = torch.stack([p.emotional_valence for p in points])
            
            # Use UMAP for visualization
            import umap
            reducer = umap.UMAP()
            embeddings_2d = reducer.fit_transform(embeddings.cpu().numpy())
            
            # Plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            # Scatter plot with uncertainty as size and emotion as color
            plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                s=uncertainties.cpu().numpy() * 100,
                c=emotional.cpu().numpy(),
                cmap='RdYlBu',
                alpha=0.6
            )
            
            # Add labels
            for i, expr in enumerate(expressions):
                plt.annotate(expr, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
                
            plt.colorbar(label='Emotional Valence')
            plt.title('Temporal-Emotional Manifold')
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
