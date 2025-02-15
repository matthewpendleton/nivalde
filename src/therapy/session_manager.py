"""Session manager for integrating system components.

This module manages the integration between BERT contextualization,
memory system, and emotional processing.
"""

import torch
from typing import Dict, List, Optional, Tuple

from ..input.bert_contextualizer import BERTContextualizer
from ..memory.memory_system import Transformer2Memory
from ..memory.emotional_embedding import EmotionalProcessor

class TherapySession:
    """Manages a therapy session, integrating all system components."""
    
    def __init__(self):
        """Initialize therapy session components."""
        self.contextualizer = BERTContextualizer()
        self.memory = Transformer2Memory()
        self.emotional_processor = EmotionalProcessor()
        
    def process_input(self,
                     text: str,
                     emotional_keywords: Optional[List[str]] = None,
                     store_memory: bool = True) -> Dict[str, torch.Tensor]:
        """Process client input through the system pipeline.
        
        Args:
            text: Client input text
            emotional_keywords: Optional emotional keywords to emphasize
            store_memory: Whether to store the processed state in memory
            
        Returns:
            Dict containing:
                - bert_context: BERT contextualized representation
                - memory_context: Historical context from memory
                - emotional_state: Current emotional state
                - processed_context: Processed combined context
        """
        # 1. Contextualize input
        bert_context = self.contextualizer(
            text,
            emotional_keywords=emotional_keywords
        )
        
        # 2. Get historical context
        memory_context = self.memory.get_historical_context(bert_context)
        
        # 3. Process emotional state
        emotional_state, processed_context = self.emotional_processor(
            bert_context,
            memory_context
        )
        
        # 4. Store in memory if requested
        if store_memory:
            self.memory.store_memory(emotional_state)
            
        return {
            'bert_context': bert_context,
            'memory_context': memory_context,
            'emotional_state': emotional_state,
            'processed_context': processed_context
        }
        
    def process_session(self,
                       conversation: List[str],
                       metadata: Optional[List[Dict]] = None) -> List[Dict]:
        """Process a complete conversation, maintaining context.
        
        Args:
            conversation: List of client utterances
            metadata: Optional list of metadata dicts with emotional_keywords
            
        Returns:
            List of state dicts for each utterance
        """
        states = []
        
        for i, text in enumerate(conversation):
            # Get emotional keywords from metadata if available
            keywords = None
            if metadata and i < len(metadata):
                keywords = metadata[i].get('emotional_keywords')
                
            # Process through pipeline
            state = self.process_input(
                text,
                emotional_keywords=keywords
            )
            states.append(state)
            
        return states
        
    def get_session_summary(self) -> Dict[str, List[float]]:
        """Get summary statistics for the current session.
        
        Returns:
            Dict containing:
                - surprise_scores: List of memory surprise scores
                - emotional_continuity: List of emotional state similarities
        """
        return {
            'surprise_scores': self.memory.surprise_scores,
            'emotional_continuity': [
                torch.cosine_similarity(
                    self.memory.memories[i],
                    self.memory.memories[i-1],
                    dim=0
                ).item()
                for i in range(1, len(self.memory.memories))
            ] if len(self.memory.memories) > 1 else []
        }
