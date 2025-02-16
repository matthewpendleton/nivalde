# generate_architecture.py
from graphviz import Digraph
import os

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'source/_static/diagrams')
os.makedirs(output_dir, exist_ok=True)

dot = Digraph(comment='Nivalde Architecture', format='png')
dot.attr(rankdir='TB')  # Top to bottom layout
dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', margin='0.3,0.2')
dot.attr(splines='ortho')  # Orthogonal edges for cleaner look

# Define node styles
MAIN_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'lightgray'}
MEMORY_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9'}  # Light green
MATH_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD'}    # Light blue
PROCESS_NODE = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF3E0'} # Light orange

# Input and BERT processing
with dot.subgraph(name='cluster_input') as inp:
    inp.attr(label='Input Processing', style='dashed')
    inp.node('input', 'Multimodal Input\n(Audio/Video/Text)', **MAIN_NODE)
    inp.node('bert', 'BERT Contextualizer\n(Emotional Enhancement)', **MAIN_NODE)
    inp.edge('input', 'bert')

# Memory system
with dot.subgraph(name='cluster_memory') as mem:
    mem.attr(label='Memory System', style='dashed')
    mem.node('t2', 'Transformer²\nProcessor', **MEMORY_NODE)
    mem.node('em', 'Episodic\nMemory', **MEMORY_NODE)
    mem.node('sm', 'Semantic\nMemory', **MEMORY_NODE)
    mem.node('pm', 'Procedural\nMemory', **MEMORY_NODE)
    
    # Memory connections
    mem.edge('t2', 'em', dir='both')
    mem.edge('t2', 'sm', dir='both')
    mem.edge('t2', 'pm', dir='both')
    mem.edge('em', 'sm', style='dotted')
    mem.edge('sm', 'pm', style='dotted')

# Mathematical foundations
with dot.subgraph(name='cluster_math') as math:
    math.attr(label='Mathematical Foundations', style='dashed')
    math.node('ees', 'Emotional Embedding Space\n(EES)', **MATH_NODE)
    math.node('map', 'State Mapping Function\nf: ℋ(EES) → TSM', **MATH_NODE)
    math.node('tsm', 'Therapeutic State Manifold\n(TSM)', **MATH_NODE)
    
    # Connect mathematical components
    math.edge('ees', 'map')
    math.edge('map', 'tsm')

# Response generation
with dot.subgraph(name='cluster_response') as resp:
    resp.attr(label='Response Generation', style='dashed')
    resp.node('rl', 'RL Optimization\n(PPO)', **PROCESS_NODE)
    resp.node('gpt', 'Response Generation\n(Contextual GPT)', **PROCESS_NODE)
    resp.node('test', 'Adversarial Testing\n(GPT-4)', **PROCESS_NODE)
    
    # Response pipeline
    resp.edge('rl', 'gpt')
    resp.edge('gpt', 'test')

# Main connections between components
dot.edge('bert', 'ees', xlabel='Emotional\nContext')  # BERT directly to EES
dot.edge('bert', 't2', xlabel='Contextual\nInformation')  # BERT also feeds T²
dot.edge('t2', 'ees', xlabel='Memory\nContext')  # T² also connects to EES
dot.edge('tsm', 'rl', xlabel='Therapeutic\nStrategy')

# Feedback loops
dot.edge('test', 'input', xlabel='Client\nFeedback', style='dashed', constraint='false')
dot.edge('test', 't2', xlabel='Memory\nUpdate', style='dashed', constraint='false')
dot.edge('rl', 'tsm', xlabel='Policy\nUpdate', style='dotted', constraint='false')

# Graph attributes for better layout
dot.attr(ranksep='1.0')
dot.attr(nodesep='0.75')
dot.attr(concentrate='true')

# Save to the correct location
output_path = os.path.join(output_dir, 'system_architecture')
dot.render(output_path, format='png', cleanup=True)
