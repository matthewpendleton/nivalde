<p align="center">
   <img src = "logo_fixed.png" 
      width = 200
      height = 200
      />
</p>

# AI Psychotherapy System

An advanced AI system for providing real-time psychotherapeutic services using multimodal input processing, emotional embedding spaces, and adaptive therapeutic interventions.

## Architecture Overview

1. **Multimodal Input Processing**
   - Direct audio-to-embedding conversion
   - Video processing
   - Text processing for journaling

2. **Contextual Processing (GPT1)**
   - Lightweight GPT model for initial context processing
   - Paralinguistic feature preservation

3. **Memory and Emotional Systems**
   - Transformer^2 based memory system
   - Hysteretic Emotional Embedding Space (EES)
   - Phase transition detection for therapeutic opportunities

4. **Therapeutic Planning**
   - Latent Therapy Manifold (LTM)
   - Unsupervised therapeutic technique emergence
   - Diffusion-based intervention planning

5. **Optimization and Execution**
   - RL-based optimization
   - GPT2 for natural language generation
   - Avatar-based therapy delivery

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`
  - `input/` - Multimodal input processing
  - `models/` - Neural network models
  - `memory/` - Memory and emotional embedding systems
  - `therapy/` - Therapeutic planning and execution
  - `avatar/` - Avatar interface
  - `utils/` - Utility functions
