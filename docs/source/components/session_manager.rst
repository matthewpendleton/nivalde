Session Manager
==============

The Session Manager integrates all system components to provide a cohesive therapy
experience.

Overview
--------

The Session Manager coordinates between the BERT Contextualizer, Memory System, and
Emotional Processor to handle complete therapy sessions.

.. code-block:: python

   from nivalde.therapy.session_manager import TherapySession
   
   session = TherapySession()
   state = session.process_input(
       "I've been feeling anxious",
       emotional_keywords=["anxious", "worry"]
   )

Key Features
-----------

1. Component Integration
   - Coordinates all system components
   - Maintains session state
   - Provides unified interface

2. Session Management
   - Processes complete conversations
   - Tracks emotional continuity
   - Generates session summaries

Implementation Details
-------------------

Processing Pipeline
~~~~~~~~~~~~~~~~

The session manager implements a complete processing pipeline:

1. Input Contextualization
   .. code-block:: python
   
      # BERT contextualization
      bert_context = contextualizer(
          text,
          emotional_keywords=emotional_keywords
      )

2. Memory Integration
   .. code-block:: python
   
      # Get historical context
      memory_context = memory.get_historical_context(bert_context)
      
      # Store new state
      memory.store_memory(emotional_state)

3. Emotional Processing
   .. code-block:: python
   
      # Process emotional state
      emotional_state, processed_context = emotional_processor(
          bert_context,
          memory_context
      )

Session Analysis
-------------

The session manager provides tools for analyzing therapy sessions:

.. code-block:: python

   # Get session summary
   summary = session.get_session_summary()
   
   # Access statistics
   surprise_scores = summary['surprise_scores']
   emotional_continuity = summary['emotional_continuity']

Usage Examples
------------

Single Input Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   session = TherapySession()
   
   # Process single input
   state = session.process_input(
       "I feel overwhelmed",
       emotional_keywords=["stress", "anxiety"]
   )
   
   # Access state components
   bert_context = state['bert_context']
   emotional_state = state['emotional_state']

Complete Conversation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   conversation = [
       "I've been feeling down lately",
       "Work has been very stressful",
       "But my family helps me cope"
   ]
   
   metadata = [
       {"emotional_keywords": ["sad", "depression"]},
       {"emotional_keywords": ["stress", "pressure"]},
       {"emotional_keywords": ["support", "cope"]}
   ]
   
   # Process complete conversation
   states = session.process_session(conversation, metadata)

Session Analysis
~~~~~~~~~~~~~

.. code-block:: python

   # Get session summary
   summary = session.get_session_summary()
   
   # Analyze emotional progression
   for i, continuity in enumerate(summary['emotional_continuity']):
       print(f"Emotional continuity at step {i}: {continuity}")
       
   # Analyze memory formation
   for i, surprise in enumerate(summary['surprise_scores']):
       print(f"Memory surprise at step {i}: {surprise}")

Best Practices
------------

1. Session Management
   - Create new session for each client
   - Maintain session across multiple interactions
   - Store session summaries for analysis

2. Emotional Keywords
   - Use relevant emotional keywords
   - Update keywords based on context
   - Include both positive and negative emotions

3. Memory Storage
   - Store all significant interactions
   - Monitor surprise scores for patterns
   - Track emotional continuity
