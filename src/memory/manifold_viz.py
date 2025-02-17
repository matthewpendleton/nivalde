"""Visualization tools for the temporal-emotional manifold."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import torch
import umap
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class ManifoldVisualizer:
    def __init__(self, manifold):
        self.manifold = manifold
        
    def encode_expressions(self, expressions: List[str]):
        """Encode expressions and extract relevant features."""
        points = self.manifold(expressions)
        
        embeddings = torch.stack([p.embedding for p in points])
        uncertainties = torch.stack([p.uncertainty for p in points])
        emotions = torch.stack([p.emotional_valence for p in points])
        
        return embeddings, uncertainties, emotions
        
    def reduce_dimensions(self, embeddings: torch.Tensor, method: str = 'umap'):
        """Reduce high-dimensional embeddings to 2D for visualization."""
        emb_np = embeddings.detach().cpu().numpy()
        
        if method == 'umap':
            reducer = umap.UMAP(
                n_components=2,
                metric='cosine',
                n_neighbors=15,
                min_dist=0.1,
                random_state=42
            )
        elif method == 'tsne':
            reducer = TSNE(
                n_components=2,
                perplexity=30,
                metric='cosine',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown dimension reduction method: {method}")
            
        return reducer.fit_transform(emb_np)
        
    def plot_temporal_flow(
        self,
        expressions: List[str],
        timestamps: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ):
        """Visualize the flow of time in the manifold.
        
        Creates an interactive plot showing:
        - Temporal progression (position)
        - Emotional valence (color)
        - Uncertainty (size)
        - Expression text (hover)
        """
        embeddings, uncertainties, emotions = self.encode_expressions(expressions)
        coords_2d = self.reduce_dimensions(embeddings)
        
        # Create temporal flow visualization
        fig = go.Figure()
        
        # Add points
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers+text',
            marker=dict(
                size=uncertainties.detach().cpu().numpy() * 50 + 10,
                color=emotions.detach().cpu().numpy(),
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Emotional Valence")
            ),
            text=expressions,
            hovertemplate="<b>Expression:</b> %{text}<br>" +
                         "<b>Uncertainty:</b> %{marker.size}<br>" +
                         "<b>Emotion:</b> %{marker.color}<br>" +
                         "<extra></extra>"
        ))
        
        # Add flow arrows between consecutive points
        for i in range(len(coords_2d)-1):
            fig.add_trace(go.Scatter(
                x=coords_2d[i:i+2, 0],
                y=coords_2d[i:i+2, 1],
                mode='lines',
                line=dict(
                    color='rgba(100,100,100,0.2)',
                    width=1,
                    dash='dot'
                ),
                showlegend=False
            ))
            
        fig.update_layout(
            title="Temporal-Emotional Flow",
            xaxis_title="Manifold Dimension 1",
            yaxis_title="Manifold Dimension 2",
            showlegend=False,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
        return fig
        
    def plot_uncertainty_landscape(
        self,
        expressions: List[str],
        grid_size: int = 20,
        save_path: Optional[str] = None
    ):
        """Visualize the uncertainty landscape of the manifold.
        
        Creates a contour plot showing:
        - Base points (scatter)
        - Interpolated uncertainty (contour)
        - Emotional valence (color)
        """
        embeddings, uncertainties, emotions = self.encode_expressions(expressions)
        coords_2d = self.reduce_dimensions(embeddings)
        
        # Create grid for interpolation
        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        
        x = np.linspace(x_min - margin, x_max + margin, grid_size)
        y = np.linspace(y_min - margin, y_max + margin, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate uncertainty
        from scipy.interpolate import griddata
        uncertainty_grid = griddata(
            coords_2d,
            uncertainties.detach().cpu().numpy(),
            (X, Y),
            method='cubic',
            fill_value=uncertainties.mean().item()
        )
        
        # Create visualization
        fig = go.Figure()
        
        # Add uncertainty contour
        fig.add_trace(go.Contour(
            x=x,
            y=y,
            z=uncertainty_grid,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Uncertainty"),
            contours=dict(
                coloring='heatmap',
                showlabels=True
            )
        ))
        
        # Add points
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers+text',
            marker=dict(
                size=15,
                color=emotions.detach().cpu().numpy(),
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Emotional Valence")
            ),
            text=expressions,
            hovertemplate="<b>Expression:</b> %{text}<br>" +
                         "<b>Emotion:</b> %{marker.color}<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title="Uncertainty Landscape",
            xaxis_title="Manifold Dimension 1",
            yaxis_title="Manifold Dimension 2",
            showlegend=False,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
        return fig
        
    def plot_emotional_clusters(
        self,
        expressions: List[str],
        n_clusters: int = 5,
        save_path: Optional[str] = None
    ):
        """Visualize emotional clustering in the manifold.
        
        Creates a scatter plot showing:
        - Emotional clusters (color)
        - Temporal distance (position)
        - Expression similarity (lines)
        """
        embeddings, uncertainties, emotions = self.encode_expressions(expressions)
        coords_2d = self.reduce_dimensions(embeddings)
        
        # Cluster emotions
        from sklearn.cluster import KMeans
        emotion_clusters = KMeans(
            n_clusters=n_clusters,
            random_state=42
        ).fit_predict(emotions.detach().cpu().numpy().reshape(-1, 1))
        
        # Create visualization
        fig = go.Figure()
        
        # Add similarity edges between points
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings.detach().cpu().numpy())
        threshold = np.percentile(sim_matrix, 90)  # Only show top 10% similarities
        
        for i in range(len(sim_matrix)):
            for j in range(i+1, len(sim_matrix)):
                if sim_matrix[i,j] > threshold:
                    fig.add_trace(go.Scatter(
                        x=coords_2d[[i,j], 0],
                        y=coords_2d[[i,j], 1],
                        mode='lines',
                        line=dict(
                            width=sim_matrix[i,j] * 2,
                            color='rgba(100,100,100,0.1)'
                        ),
                        showlegend=False
                    ))
        
        # Add points
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers+text',
            marker=dict(
                size=15,
                color=emotion_clusters,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Emotional Cluster")
            ),
            text=expressions,
            hovertemplate="<b>Expression:</b> %{text}<br>" +
                         "<b>Cluster:</b> %{marker.color}<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title="Emotional Clusters",
            xaxis_title="Manifold Dimension 1",
            yaxis_title="Manifold Dimension 2",
            showlegend=False,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
        return fig
        
    def create_interactive_dashboard(
        self,
        expressions: List[str],
        save_path: Optional[str] = None
    ):
        """Create an interactive dashboard with multiple views of the manifold."""
        # Create all three visualizations
        flow_fig = self.plot_temporal_flow(expressions)
        uncertainty_fig = self.plot_uncertainty_landscape(expressions)
        clusters_fig = self.plot_emotional_clusters(expressions)
        
        # Combine into dashboard
        dashboard = go.Figure()
        
        # Add subplot for each visualization
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Temporal Flow",
                "Uncertainty Landscape",
                "Emotional Clusters"
            )
        )
        
        # Add traces from individual plots
        for trace in flow_fig.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in uncertainty_fig.data:
            fig.add_trace(trace, row=1, col=2)
        for trace in clusters_fig.data:
            fig.add_trace(trace, row=2, col=1)
            
        fig.update_layout(
            height=1000,
            title_text="Temporal-Emotional Manifold Analysis",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        return fig
