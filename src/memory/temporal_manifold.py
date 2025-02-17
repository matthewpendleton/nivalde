"""Temporal-emotional manifold learning system."""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from dataclasses import dataclass
from typing import List, Optional
import torch.nn.functional as F

@dataclass
class ManifoldPoint:
    """A point in the temporal-emotional manifold."""
    embedding: torch.Tensor
    emotional_valence: torch.Tensor
    uncertainty: torch.Tensor

class TemporalManifold(nn.Module):
    """Neural manifold for temporal-emotional memory representation."""
    
    def __init__(
        self,
        manifold_dim: int = 64,
        emotional_dim: int = 16,
        n_heads: int = 8,
        model_name: str = "bert-base-uncased"
    ):
        super().__init__()
        
        # Initialize BERT components
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Projection layers
        bert_dim = self.bert.config.hidden_size
        self.temporal_proj = nn.Sequential(
            nn.Linear(bert_dim, manifold_dim * 2),
            nn.ReLU(),
            nn.Linear(manifold_dim * 2, manifold_dim)
        )
        
        self.emotional_proj = nn.Sequential(
            nn.Linear(bert_dim, emotional_dim * 2),
            nn.ReLU(),
            nn.Linear(emotional_dim * 2, 1),
            nn.Tanh()
        )
        
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(bert_dim, emotional_dim),
            nn.ReLU(),
            nn.Linear(emotional_dim, 1),
            nn.Sigmoid()
        )
        
        # Context attention
        self.context_attention = nn.MultiheadAttention(
            embed_dim=manifold_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Context integration
        self.context_gate = nn.Sequential(
            nn.Linear(manifold_dim * 2, manifold_dim),
            nn.Sigmoid()
        )
        
    def _get_bert_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for text."""
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.bert(**tokens)
            
        # Use [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :]
        
    def forward(
        self,
        expressions: List[str],
        context_expressions: Optional[List[List[str]]] = None,
        context_timestamps: Optional[torch.Tensor] = None
    ) -> List[ManifoldPoint]:
        """
        Forward pass through the manifold.
        
        Args:
            expressions: List of temporal expressions
            context_expressions: Optional list of context expressions for each input
            context_timestamps: Optional tensor of context timestamps
            
        Returns:
            List of ManifoldPoint containing embedding and metadata
        """
        # Get BERT embeddings for main expressions
        embeddings = torch.cat([
            self._get_bert_embedding(expr) for expr in expressions
        ])
        
        # Project to manifold space
        temporal_features = self.temporal_proj(embeddings)
        
        # If we have context, use attention to integrate it
        if context_expressions is not None and context_timestamps is not None:
            # Get context embeddings
            context_embs = []
            for expr_list in context_expressions:
                # Handle each expression's context
                ctx_embs = torch.cat([
                    self._get_bert_embedding(ctx) for ctx in expr_list
                ])
                ctx_embs = self.temporal_proj(ctx_embs)
                context_embs.append(ctx_embs)
            
            # Stack into batch
            context_embs = torch.stack(context_embs)
            
            # Apply attention
            attended_ctx, _ = self.context_attention(
                temporal_features.unsqueeze(1),
                context_embs,
                context_embs
            )
            attended_ctx = attended_ctx.squeeze(1)
            
            # Compute context integration gate
            gate = self.context_gate(
                torch.cat([temporal_features, attended_ctx], dim=-1)
            )
            
            # Update temporal features
            temporal_features = (
                gate * temporal_features +
                (1 - gate) * attended_ctx
            )
        
        # Project to emotional space
        emotional_valence = self.emotional_proj(embeddings)
        
        # Compute uncertainty
        uncertainty = self.uncertainty_proj(embeddings)
        
        # Create manifold points
        points = []
        for i in range(len(expressions)):
            points.append(ManifoldPoint(
                embedding=temporal_features[i],
                emotional_valence=emotional_valence[i],
                uncertainty=uncertainty[i]
            ))
            
        return points

class TemporalLoss(nn.Module):
    """Loss function for training the temporal manifold."""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        points: List[ManifoldPoint],
        temporal_order: torch.Tensor,  # Ground truth temporal ordering
        emotional_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss for a sequence of manifold points."""
        
        # Extract embeddings and stack
        embeddings = torch.stack([p.embedding for p in points])
        
        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings)
        
        # Compute ground truth distances
        true_dists = torch.abs(
            temporal_order.unsqueeze(1) - temporal_order.unsqueeze(0)
        )
        
        # Temporal ordering loss
        temporal_loss = F.mse_loss(dists, true_dists)
        
        # Uncertainty calibration
        uncertainties = torch.stack([p.uncertainty for p in points])
        target_uncertainty = torch.mean(true_dists, dim=1, keepdim=True)
        uncertainty_loss = F.mse_loss(uncertainties, target_uncertainty)
        
        # Emotional alignment if labels provided
        emotional_loss = 0.0
        if emotional_labels is not None:
            emotional_preds = torch.stack([p.emotional_valence for p in points])
            emotional_loss = F.mse_loss(emotional_preds, emotional_labels)
            
        # Combine losses with weights
        total_loss = (
            temporal_loss +
            0.5 * uncertainty_loss +
            0.3 * emotional_loss
        )
        
        return total_loss
