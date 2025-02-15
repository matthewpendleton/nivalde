"""Emotional Embedding Space implementation using Neural ODEs.

This module implements the continuous-time emotional state representation using
Neural Ordinary Differential Equations with hysteretic memory gates.

References:
    [1] Chen et al. "Neural Ordinary Differential Equations" (2018)
    [2] Chang et al. "Antisymmetric Neural Networks" (2023)
"""

import torch
import torch.nn as nn
import torchdiffeq

class EmotionalEmbeddingODE(nn.Module):
    """Neural ODE implementation of emotional state dynamics.
    
    Attributes:
        dim (int): Dimension of the emotional embedding space
        dynamics_net (nn.Module): Neural network modeling the intrinsic dynamics
        memory_net (nn.Module): Neural network for memory influence
    
    Mathematical formulation:
        The emotional state evolution follows:
        dE/dt = f_θ(E, t) + g_φ(M, t)
        
        Where:
            - E is the emotional state vector
            - f_θ is the intrinsic dynamics network
            - g_φ is the memory influence network
            - M is the memory state
    """
    
    def __init__(self, dim: int = 768):
        """Initialize the ODE system.
        
        Args:
            dim: Dimension of the emotional embedding space
        """
        super().__init__()
        self.dim = dim
        self.dynamics_net = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.Tanh(),
            nn.Linear(2*dim, dim)
        )
        self.memory_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh()
        )
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute the time derivative of the emotional state.
        
        Args:
            t: Current time point
            state: Current emotional state
            
        Returns:
            Time derivative dE/dt
        """
        dE_dt = self.dynamics_net(state) + self.memory_net(state)
        return dE_dt

class HystereticGate(nn.Module):
    """Hysteretic gate for controlling information flow in emotional updates.
    
    Mathematical formulation:
        Γ(E, M) = σ(W_γ [E; M] + b_γ)
        
    Where:
        - E is the current emotional state
        - M is the memory state
        - σ is the sigmoid activation
        - W_γ, b_γ are learnable parameters
    """
    
    def __init__(self, in_features: int, persistence_factor: float = 0.8):
        """Initialize the hysteretic gate.
        
        Args:
            in_features: Dimension of input features
            persistence_factor: Base persistence rate for emotional states
        """
        super().__init__()
        self.gate = nn.Linear(2*in_features, in_features)
        self.persistence_factor = persistence_factor
    
    def forward(self, emotional_state: torch.Tensor, memory_state: torch.Tensor) -> torch.Tensor:
        """Compute the gating values.
        
        Args:
            emotional_state: Current emotional state
            memory_state: Current memory state
            
        Returns:
            Gating values in [0,1]
        """
        combined = torch.cat([emotional_state, memory_state], dim=-1)
        return torch.sigmoid(self.gate(combined)) * self.persistence_factor

class EmotionalEmbeddingSpace(nn.Module):
    """Complete emotional embedding space implementation.
    
    This class combines the Neural ODE dynamics with hysteretic gating and
    phase transition detection to create a complete emotional representation
    system.
    
    Attributes:
        dim (int): Dimension of the emotional embedding space
        ode (callable): ODE solver function
        dynamics (EmotionalEmbeddingODE): ODE system for emotional dynamics
        memory_gate (HystereticGate): Gate for memory persistence
        phase_detector (PhaseTransitionDetector): Detector for emotional phase transitions
    """
    
    def __init__(self, dim: int = 768, memory_size: int = 512):
        """Initialize the emotional embedding space.
        
        Args:
            dim: Dimension of the emotional embedding space
            memory_size: Size of the memory state
        """
        super().__init__()
        self.dim = dim
        self.ode = torchdiffeq.odeint_adjoint
        self.dynamics = EmotionalEmbeddingODE(dim)
        self.memory_gate = HystereticGate(dim)
        self.phase_detector = PhaseTransitionDetector(
            threshold=0.8,
            window_size=20
        )
    
    def forward(self, x: torch.Tensor, t_span: torch.Tensor) -> tuple:
        """Evolve the emotional state through time.
        
        Args:
            x: Initial emotional state
            t_span: Time points to solve for
            
        Returns:
            tuple: (emotional_trajectory, phase_transitions)
                - emotional_trajectory: Tensor of shape [time, batch, dim]
                - phase_transitions: Detected phase transition points
        """
        trajectory = self.ode(
            self.dynamics,
            x,
            t_span,
            method='dopri5',
            adjoint_method='adjoint_midpoint'
        )
        transitions = self.phase_detector(trajectory)
        return trajectory, transitions
