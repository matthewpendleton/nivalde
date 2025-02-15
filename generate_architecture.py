# Updated generate_architecture.py
from graphviz import Digraph

dot = Digraph(comment='Enhanced Architecture', format='png')
dot.attr(rankdir='TB', splines='ortho')

# Main Components
dot.node('A', 'Multimodal Input\n(Audio/Video/Text)')
dot.node('B', 'Contextual Processor\n(BERT)')
dot.node('C', 'Emotional Embedding\n(Transformer²)')
dot.node('D', 'Long-Term Memory\n(Hysteretic Gates)')
dot.node('E', 'Therapy Manifold\n(Diffusion)')
dot.node('F', 'RL Optimization\n(PPO)') 
dot.node('G', 'Contextual GPT\n(Interpreter)')
dot.node('H', 'Adversarial Testing\n(GPT-4)')

# Data Flows
dot.edges([
    ('A', 'B'),
    ('B', 'C'),
    ('C', 'D'),
    ('D', 'E'),
    ('E', 'F'),
    ('F', 'G'),
    ('G', 'H')
])

# Feedback Loops
dot.edge('H', 'A', xlabel='Client Feedback', style='dashed')
dot.edge('H', 'D', xlabel='Memory Update', style='dashed')
dot.edge('F', 'C', xlabel='Embedding Tuning', style='dotted')

# Subsystems
with dot.subgraph(name='cluster_memory') as c:
    c.attr(style='dashed', label='Memory System')
    c.node('D1', 'Episodic Memory')
    c.node('D2', 'Semantic Memory')
    c.node('D3', 'Procedural Memory')
    c.edges([('D1', 'D2'), ('D2', 'D3')])
    dot.edge('D', 'D1', style='invis')

dot.render('docs/architecture-crossplatform', cleanup=True)
