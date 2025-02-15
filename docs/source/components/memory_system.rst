Memory System
=============

The Nivalde AI Platform uses a standard implementation of the Transformer² architecture
for hierarchical memory storage and retrieval.

Overview
--------

The memory system stores client memories hierarchically based on their novelty/surprise,
providing historical context to the Emotional Embedding Space (EES).

.. code-block:: python

   from nivalde.memory.memory_system import Transformer2Memory
   
   memory = Transformer2Memory()
   memory.store_memory(new_memory)
   historical_context = memory.get_historical_context(current_input)

Key Features
-----------

1. Hierarchical Storage
   - Memories stored based on surprise/novelty
   - Natural emergence of memory hierarchy
   - Efficient retrieval of relevant context

2. Surprise Computation
   - Based on negative log probability
   - Considers prior memory context
   - Adapts to client's emotional patterns

Implementation Details
-------------------

Memory Storage
~~~~~~~~~~~~

New memories are stored hierarchically based on their surprise score:

.. code-block:: python

   def compute_surprise(memory, prior_memories):
       """Compute surprise score for new memory."""
       return -log_probability(memory | prior_memories)

The Transformer² architecture naturally handles this hierarchical organization through
its attention mechanisms and positional encodings.

Memory Retrieval
~~~~~~~~~~~~~

When processing new client input, relevant historical context is retrieved:

.. code-block:: python

   def get_historical_context(current_input):
       """Retrieve relevant historical context."""
       return transformer(
           torch.cat([current_input, memories])
       )[0]  # First token has context

Integration with EES
-----------------

The memory system provides historical context to the EES in two ways:

1. Direct Memory Access
   - Current session memories
   - Recent historical context
   - Relevant past experiences

2. Contextualized History
   - Processed through attention mechanisms
   - Weighted by relevance to current state
   - Filtered for therapeutic significance

Usage Examples
------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   memory = Transformer2Memory()
   
   # Store new memory
   memory.store_memory(new_memory)
   
   # Get historical context
   context = memory.get_historical_context(current_input)

Memory Analysis
~~~~~~~~~~~~

.. code-block:: python

   # Get surprise scores
   scores = memory.surprise_scores
   
   # Analyze memory hierarchy
   for memory, score in zip(memory.memories, scores):
       print(f"Memory surprise: {score}")

Configuration
-----------

The memory system can be configured with different parameters:

.. code-block:: python

   # Default configuration
   memory = Transformer2Memory()
   
   # Custom configuration
   memory = Transformer2Memory(
       dim=512,
       num_layers=6,
       num_heads=8
   )
