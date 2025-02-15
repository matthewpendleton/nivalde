# Mathematical Foundations

## Emotional Embedding Space (EES)

### Neural ODE Formulation

The emotional state evolution in Nivalde is modeled using a Neural Ordinary Differential Equation (Neural ODE) system with hysteretic memory gates. This approach allows for continuous-time modeling of emotional dynamics while maintaining stability and interpretability.

#### Core Equations

The emotional state evolution is governed by:

$$
\frac{d\mathcal{E}}{dt} = f_\theta(\mathcal{E}, t) + g_\phi(\mathcal{M}, t)
$$

Where:
- $\mathcal{E}$ represents the emotional state vector
- $f_\theta$ is a neural network modeling intrinsic dynamics
- $g_\phi$ is the memory influence function
- $\mathcal{M}$ represents the memory state

#### Hysteretic Gate Mechanism

The hysteretic gate $\Gamma$ controls information flow:

$$
\Gamma(E, M) = \sigma\left(W_\gamma [E; M] + b_\gamma\right)
$$

Where:
- $E$ is the current emotional state
- $M$ is the memory state
- $\sigma$ is the sigmoid activation
- $W_\gamma, b_\gamma$ are learnable parameters

#### State Update Rule

The discrete-time approximation of the system:

$$
\mathcal{E}_t = \Gamma \odot \mathcal{E}_{t-1} + (1 - \Gamma) \odot \text{Tanh}\left(\sum_{k=1}^K \alpha_k \cdot \text{MultiHeadAttn}(Q_k, K_{1:N}, V_{1:N})\right)
$$

#### Stability Conditions

The system maintains stability through a Lyapunov function:

$$
V(\mathcal{E}) = \frac{1}{2}\|\mathcal{E}\|^2_2 \quad \text{s.t.} \quad \frac{dV}{dt} \leq 0
$$

## Therapy Manifold

### Continuous Response Space

The therapy manifold is modeled as a Riemannian manifold with metric tensor $G(x)$:

$$
ds^2 = \sum_{i,j} G_{ij}(x)dx^idx^j
$$

### Intervention Generation

Interventions are sampled using a diffusion process:

$$
dx_t = -\frac{1}{2}\nabla U(x_t)dt + \sqrt{\beta_t}dW_t
$$

Where:
- $U(x)$ is the potential energy function
- $\beta_t$ is the temperature schedule
- $W_t$ is a Wiener process

## Memory System

### Transformer² Architecture

The memory system uses a dual-stack attention mechanism:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

With hysteretic updates:

$$
M_t = \lambda_t M_{t-1} + (1-\lambda_t) \tilde{M}_t
$$

Where $\lambda_t$ is a learned update gate.
