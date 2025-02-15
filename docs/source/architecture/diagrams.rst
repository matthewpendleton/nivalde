System Diagrams
==============

This section provides visual representations of the Nivalde AI Platform's architecture
and component interactions.

System Architecture
-----------------

The following diagram shows the high-level system architecture and component interactions:

.. graphviz:: ../_static/diagrams/system_architecture.dot

Key Components:

1. Input Processing (Blue)
   - Client input processing
   - BERT contextualization
   - Emotional keyword emphasis

2. Memory System (Green)
   - Transformer² architecture
   - Hierarchical storage
   - Surprise-based organization

3. Emotional Processing (Red)
   - Context integration
   - State evolution
   - Natural hysteresis

Memory System
------------

This diagram details the memory system's hierarchical storage and retrieval:

.. graphviz:: ../_static/diagrams/memory_system.dot

Key Features:

1. Memory Input
   - New memory processing
   - Surprise computation
   - Score-based organization

2. Hierarchical Storage
   - Surprise-based hierarchy
   - Memory categorization
   - Efficient organization

3. Context Retrieval
   - Transformer attention
   - Relevant context selection
   - Historical context generation

Emotional Processing
------------------

The emotional processing pipeline is illustrated here:

.. graphviz:: ../_static/diagrams/emotional_processing.dot

Key Components:

1. Input Context
   - BERT-contextualized input
   - Historical memory context
   - Context combination

2. Context Processing
   - Combined context processing
   - Feature extraction
   - Emotional emphasis

3. State Integration
   - Previous state influence
   - Natural hysteresis
   - State evolution

Data Flow
--------

The system processes client input through several stages:

1. Input Processing
   - Client input → BERT Contextualizer
   - Emotional keyword emphasis
   - Contextualized representation

2. Memory Integration
   - Surprise computation
   - Hierarchical storage
   - Context retrieval

3. Emotional Processing
   - Context combination
   - State integration
   - Emotional evolution

Component Interactions
--------------------

The components interact in a cyclical manner:

1. Forward Flow
   - Input → BERT → Memory
   - Memory → Emotional Processor
   - Processor → New State

2. Feedback Loop
   - New State → Memory Storage
   - Memory → Future Processing
   - Natural Hysteresis

3. Integration Points
   - BERT ↔ Memory
   - Memory ↔ Processor
   - Processor ↔ State
