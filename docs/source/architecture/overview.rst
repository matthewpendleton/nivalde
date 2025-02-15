System Architecture
==================

Overview
--------

The Nivalde AI Psychotherapy Platform consists of several key components:

1. Multimodal Input Processing
2. Emotional Embedding Space (EES)
3. Memory System with Transformer²
4. Therapy Manifold
5. Response Generation

Component Interactions
--------------------

.. graphviz::

   digraph architecture {
      rankdir=TB;
      node [shape=box, style=filled, fillcolor=lightgray];
      
      subgraph cluster_0 {
         label="Input Processing";
         style=dashed;
         "Multimodal Input" -> "BERT";
         "BERT" -> "Transformer²";
      }
      
      subgraph cluster_1 {
         label="Memory System";
         style=dashed;
         "Transformer²" -> "Episodic";
         "Transformer²" -> "Semantic";
         "Transformer²" -> "Procedural";
      }
      
      subgraph cluster_2 {
         label="Embedding Spaces";
         style=dashed;
         node [shape=doubleoctagon];
         "EES";
         "Therapy Manifold";
      }
      
      "Transformer²" -> "EES";
      "EES" -> "Therapy Manifold";
      "Therapy Manifold" -> "Response";
   }
