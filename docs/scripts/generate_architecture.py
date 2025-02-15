# generate_architecture.py
from graphviz import Digraph
import os

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'source/_static/diagrams')
os.makedirs(output_dir, exist_ok=True)

dot = Digraph(comment='Nivalde Architecture', format='png')
dot.attr(rankdir='TB', splines='polyline')
dot.attr('node', shape='ellipse', style='filled', fillcolor='white')

# Input processing
with dot.subgraph(name='cluster_input') as inp:
    inp.attr(label='Input Processing', style='dashed')
    inp.node('A', 'Multimodal Input\n(Audio/Video/Text)')
    inp.node('B', 'BERT Contextualizer\n(Emotional Enhancement)')
    inp.edge('A', 'B')

# Memory and Transformer² system
with dot.subgraph(name='cluster_memory') as mem:
    mem.attr(label='Memory System', style='dashed')
    # Transformer² as central memory processor
    mem.node('T2', 'Transformer²\nProcessor', shape='doubleoctagon')
    
    # Memory components
    mem.node('EM', 'Episodic Memory')
    mem.node('SM', 'Semantic Memory')
    mem.node('PM', 'Procedural Memory')
    
    # Memory hierarchy with Transformer² at center
    mem.edge('T2', 'EM', dir='both', xlabel='Write/Read')
    mem.edge('T2', 'SM', dir='both', xlabel='Write/Read')
    mem.edge('T2', 'PM', dir='both', xlabel='Write/Read')
    
    # Memory relationships
    mem.edge('EM', 'SM', style='dotted')
    mem.edge('SM', 'PM', style='dotted')

# State spaces and mapping
with dot.subgraph(name='cluster_therapeutic') as emb:
    emb.attr(label='Therapeutic Processing', style='dashed')
    emb.node('EES', 'Emotional Embedding Space\n(EES)', shape='doubleoctagon')
    emb.node('MAP', 'State Mapping Function\nf: ℋ(EES) → TSM')
    emb.node('TSM', 'Therapeutic State Manifold\n(TSM)', shape='doubleoctagon')
    emb.edge('EES', 'MAP')
    emb.edge('MAP', 'TSM')

# Output processing
dot.node('RL', 'RL Optimization\n(PPO)')
dot.node('GPT', 'Contextual GPT\n(Response Generation)')
dot.node('AT', 'Adversarial Testing\n(GPT-4)')

# Connect components
dot.edge('B', 'T2', xlabel='New Information')
dot.edge('T2', 'EES', xlabel='Memory Context')
dot.edge('TSM', 'RL', xlabel='Intervention Selection')
dot.edge('RL', 'GPT')
dot.edge('GPT', 'AT')

# Feedback loops
dot.edge('AT', 'A', xlabel='Client Feedback', style='dashed', constraint='false')
dot.edge('AT', 'T2', xlabel='Memory Update', style='dashed', constraint='false')
dot.edge('RL', 'TSM', xlabel='Policy Update', style='dotted', constraint='false')

# Graph attributes
dot.attr(ranksep='1.0')
dot.attr(nodesep='0.75')
dot.attr(concentrate='true')

# Save to the correct location
output_path = os.path.join(output_dir, 'system_architecture')
dot.render(output_path, format='png', cleanup=True)
