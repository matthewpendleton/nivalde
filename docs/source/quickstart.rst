Quickstart
==========

Basic Usage
----------

1. Initialize the system:

   .. code-block:: python

      from nivalde import NivaldeAI
      
      system = NivaldeAI()

2. Start a session:

   .. code-block:: python

      session = system.create_session()

3. Process input:

   .. code-block:: python

      response = session.process(
          "I've been feeling anxious lately",
          emotional_keywords=["anxious", "worry"]
      )

Example Session
-------------

Here's a complete therapeutic interaction:

.. code-block:: python

   from nivalde import NivaldeAI
   
   # Initialize system
   system = NivaldeAI()
   session = system.create_session()
   
   # First interaction
   response = session.process(
       "I've been feeling overwhelmed at work",
       emotional_keywords=["overwhelmed", "stress"]
   )
   print(response)  # System provides empathetic response
   
   # Follow-up
   response = session.process(
       "It's affecting my sleep",
       emotional_keywords=["worry", "sleep"]
   )
   print(response)  # System integrates previous context

For more examples, see :doc:`examples/complete_session`.
