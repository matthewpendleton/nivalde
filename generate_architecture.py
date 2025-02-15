# generate_architecture.py
from graphviz import Digraph

dot = Digraph(comment='Nivalde Architecture', format='png')
dot.attr(rankdir='TB', splines='polyline')
dot.attr('node', shape='ellipse', style='filled', fillcolor='white')

# Input processing
with dot.subgraph(name='cluster_input') as inp:
    inp.attr(label='Input Processing', style='dashed')
    inp.node('A', 'Multimodal Input\n(Audio/Video/Text)')
    inp.node('B', 'Contextual Processor\n(BERT)')
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

# Embedding spaces
with dot.subgraph(name='cluster_embeddings') as emb:
    emb.attr(label='Prototypical Embedding Spaces', style='dashed')
    emb.node('EES', 'Emotional Embedding Space\n(Hysteretic)', shape='doubleoctagon')
    emb.node('TM', 'Therapy Manifold\n(Continuous Response Space)', shape='doubleoctagon')

# Output processing
dot.node('RL', 'RL Optimization\n(PPO)')
dot.node('GPT', 'Contextual GPT\n(Response Generation)')
dot.node('AT', 'Adversarial Testing\n(GPT-4)')

# Connect components
dot.edge('B', 'T2', xlabel='New Information')
dot.edge('T2', 'EES', xlabel='Memory Context')
dot.edge('EES', 'TM', xlabel='Current State')
dot.edge('TM', 'RL', xlabel='Intervention Selection')
dot.edge('RL', 'GPT')
dot.edge('GPT', 'AT')

# Feedback loops
dot.edge('AT', 'A', xlabel='Client Feedback', style='dashed', constraint='false')
dot.edge('AT', 'T2', xlabel='Memory Update', style='dashed', constraint='false')
dot.edge('RL', 'TM', xlabel='Policy Update', style='dotted', constraint='false')

# Graph attributes
dot.attr(ranksep='1.0')
dot.attr(nodesep='0.75')
dot.attr(concentrate='true')

dot.render('docs/architecture-crossplatform', format='png', cleanup=True)
