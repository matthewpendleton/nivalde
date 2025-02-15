Complete Therapy Session
=====================

This example demonstrates a complete therapy session using the Nivalde AI Platform.

Session Setup
------------

.. code-block:: python

   from nivalde import NivaldeAI
   
   # Initialize the system
   system = NivaldeAI()
   session = system.create_session()

Initial Interaction
-----------------

.. code-block:: python

   # Client expresses anxiety about work
   response = session.process(
       "I've been feeling overwhelmed at work lately",
       emotional_keywords=["overwhelmed", "anxiety"]
   )
   
   print("Client: I've been feeling overwhelmed at work lately")
   print(f"System: {response}")

System State Analysis
------------------

After the initial interaction:

1. Emotional State
   
   .. code-block:: python
   
      print(session.get_emotional_state())
      # Output: {'anxiety': 0.8, 'stress': 0.7, 'calm': 0.2}

2. Memory Context
   
   .. code-block:: python
   
      print(session.get_memory_context())
      # Output: Shows relevant historical patterns

Follow-up Interaction
------------------

.. code-block:: python

   # Client provides more context
   response = session.process(
       "It's affecting my sleep and relationships",
       emotional_keywords=["sleep", "relationships"]
   )
   
   print("Client: It's affecting my sleep and relationships")
   print(f"System: {response}")

Therapeutic Intervention
---------------------

The system demonstrates:

1. Context Integration
   - Combines work stress with sleep issues
   - Recognizes impact on relationships

2. Emotional Evolution
   - Tracks anxiety levels
   - Maintains therapeutic consistency

3. Response Generation
   - Provides empathetic understanding
   - Offers practical coping strategies

Session Analysis
--------------

Final emotional state shows:

.. code-block:: python

   print(session.get_emotional_trajectory())
   # Output: Shows emotional state evolution

Key observations:

1. Natural emotional transitions
2. Consistent therapeutic approach
3. Context-aware responses
