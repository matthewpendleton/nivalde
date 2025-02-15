BERT Contextualizer
=================

The BERT Contextualizer processes client input to provide rich semantic and emotional context
to the system.

Overview
--------

The contextualizer uses BERT-large to create contextualized representations of client input,
with special emphasis on emotional content.

.. code-block:: python

   from nivalde.input.bert_contextualizer import BERTContextualizer
   
   contextualizer = BERTContextualizer()
   context = contextualizer(
       "I've been feeling anxious lately",
       emotional_keywords=["anxious", "worry"]
   )

Key Features
-----------

1. Emotional Emphasis
   - Attention mechanism for emotional keywords
   - Weighted contextualization based on emotional content
   - Enhanced representation of emotional states

2. Semantic Understanding
   - Rich contextual embeddings from BERT-large
   - Handles complex linguistic patterns
   - Maintains conversational context

Implementation Details
-------------------

The contextualizer processes input in several stages:

1. Tokenization
   .. code-block:: python
   
      tokenizer_output = self.tokenizer(
          text,
          padding=True,
          truncation=True,
          return_tensors="pt"
      )

2. BERT Processing
   .. code-block:: python
   
      with torch.no_grad():
          bert_output = self.bert(**tokenizer_output)
      context = bert_output.last_hidden_state[:, 0]

3. Emotional Emphasis
   .. code-block:: python
   
      # Weight context with keyword attention
      context = context * (1 + keyword_attention)

Integration with Memory
--------------------

The contextualized output feeds directly into the memory system:

.. code-block:: python

   # Process client input
   context = contextualizer(client_input)
   
   # Store in memory
   memory_system.store_memory(context)

Usage Examples
------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   contextualizer = BERTContextualizer()
   
   # Simple input
   context = contextualizer("I feel happy today")
   
   # With emotional keywords
   context = contextualizer(
       "I feel happy today",
       emotional_keywords=["happy", "joy", "positive"]
   )

Batch Processing
~~~~~~~~~~~~~

.. code-block:: python

   texts = [
       "I feel anxious",
       "Work is stressful",
       "Family helps me cope"
   ]
   
   metadata = [
       {"emotional_keywords": ["anxious", "worry"]},
       {"emotional_keywords": ["stress", "pressure"]},
       {"emotional_keywords": ["support", "comfort"]}
   ]
   
   contexts = contextualizer.batch_contextualize(texts, metadata)

Configuration
-----------

The contextualizer can be configured with different BERT models:

.. code-block:: python

   # Default (BERT-large)
   contextualizer = BERTContextualizer()
   
   # Custom BERT model
   contextualizer = BERTContextualizer(
       model_name="bert-base-uncased"
   )
