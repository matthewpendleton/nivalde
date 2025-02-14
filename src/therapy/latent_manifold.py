import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class LatentTherapyManifold(nn.Module):
    def __init__(self, 
                 embedding_dim: int,
                 latent_dim: int = 512,
                 n_diffusion_steps: int = 1000):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.n_diffusion_steps = n_diffusion_steps
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Diffusion model components
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.diffusion_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim)
            ) for _ in range(4)
        ])
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, embedding_dim)
        )
        
        # Beta schedule for diffusion
        self.beta = torch.linspace(0.0001, 0.02, n_diffusion_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
    def diffusion_forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Forward diffusion process"""
        alpha_t = self.alpha_bar[t]
        noise = torch.randn_like(x)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        
    def diffusion_reverse(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Reverse diffusion process (denoising)"""
        time_emb = self.time_embed(torch.tensor([[t/self.n_diffusion_steps]]))
        
        for layer in self.diffusion_net:
            x = layer(torch.cat([x, time_emb.repeat(x.size(0), 1)], dim=-1))
            
        return x
        
    def encode(self, emotional_state: torch.Tensor) -> torch.Tensor:
        """Encode emotional state into latent therapy manifold"""
        return self.encoder(emotional_state)
        
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent therapy manifold to intervention space"""
        return self.decoder(latent)
        
    def sample_intervention(self, 
                          emotional_state: torch.Tensor,
                          temperature: float = 1.0) -> torch.Tensor:
        """Sample therapeutic intervention using diffusion process"""
        # Encode to latent space
        latent = self.encode(emotional_state)
        
        # Add noise gradually
        x = torch.randn_like(latent) * temperature
        
        # Reverse diffusion process
        for t in reversed(range(self.n_diffusion_steps)):
            x = self.diffusion_reverse(x, t)
            
            if t > 0:  # Skip adding noise at t=0
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.beta[t]) * temperature
                x = x + sigma * noise
                
        # Decode to intervention space
        return self.decode(x)

class RLOptimizer:
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.target_critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Copy weights
        self.target_critic.load_state_dict(self.critic.state_dict())
        
    def optimize_intervention(self,
                            emotional_state: torch.Tensor,
                            proposed_intervention: torch.Tensor,
                            previous_outcomes: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """Optimize therapeutic intervention based on previous outcomes"""
        with torch.no_grad():
            state_action = torch.cat([emotional_state, proposed_intervention], dim=-1)
            baseline_value = self.target_critic(state_action)
            
        # Simple gradient-based adjustment
        for _ in range(5):  # Small number of optimization steps
            intervention = proposed_intervention.clone().requires_grad_(True)
            state_action = torch.cat([emotional_state, intervention], dim=-1)
            value = self.critic(state_action)
            
            # Maximize expected value
            loss = -value.mean()
            loss.backward()
            
            with torch.no_grad():
                intervention = intervention - 0.01 * intervention.grad
                
        return intervention

class TherapeuticPlanner:
    def __init__(self, embedding_dim: int = 768):  
        self.manifold = LatentTherapyManifold(embedding_dim)
        self.optimizer = RLOptimizer(embedding_dim, embedding_dim)
        self.previous_outcomes = []
        
    def plan_intervention(self,
                         emotional_state: torch.Tensor,
                         temperature: float = 1.0) -> torch.Tensor:
        """Plan therapeutic intervention given current emotional state"""
        # Sample initial intervention from manifold
        proposed_intervention = self.manifold.sample_intervention(
            emotional_state, temperature
        )
        
        # Optimize intervention using RL
        optimized_intervention = self.optimizer.optimize_intervention(
            emotional_state,
            proposed_intervention,
            self.previous_outcomes
        )
        
        return optimized_intervention
        
    def record_outcome(self,
                      emotional_state: torch.Tensor,
                      intervention: torch.Tensor,
                      outcome_score: float):
        """Record intervention outcome for future optimization"""
        self.previous_outcomes.append((
            torch.cat([emotional_state, intervention], dim=-1),
            outcome_score
        ))
        
        # Keep only recent outcomes
        if len(self.previous_outcomes) > 1000:
            self.previous_outcomes = self.previous_outcomes[-1000:]
