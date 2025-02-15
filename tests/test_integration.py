"""Integration tests for the Nivalde AI system."""

import unittest
import torch
from src.input.bert_contextualizer import BERTContextualizer
from src.memory.memory_system import Transformer2Memory
from src.memory.emotional_embedding import EmotionalProcessor

class TestTherapyPipeline(unittest.TestCase):
    def setUp(self):
        """Initialize system components."""
        self.contextualizer = BERTContextualizer()
        self.memory = Transformer2Memory()
        self.emotional_processor = EmotionalProcessor()
        
    def test_full_pipeline(self):
        """Test complete pipeline from input to emotional state."""
        # Test input
        text = "I've been feeling anxious about work lately."
        emotional_keywords = ["anxious", "stress", "worry"]
        
        # 1. Contextualize input
        bert_context = self.contextualizer(
            text,
            emotional_keywords=emotional_keywords
        )
        self.assertEqual(bert_context.size(-1), 768)
        
        # 2. Process through memory
        # First interaction - maximum surprise
        self.memory.store_memory(bert_context)
        self.assertAlmostEqual(self.memory.surprise_scores[0], 1.0)
        
        # Get historical context
        memory_context = self.memory.get_historical_context(bert_context)
        self.assertEqual(memory_context.size(-1), 768)
        
        # 3. Process emotional state
        emotional_state, processed_context = self.emotional_processor(
            bert_context,
            memory_context
        )
        
        # Verify dimensions
        self.assertEqual(emotional_state.size(-1), 768)
        self.assertEqual(processed_context.size(-1), 768)
        
        # 4. Test memory influence
        # Process similar input
        similar_text = "Work has been making me feel very stressed."
        similar_context = self.contextualizer(
            similar_text,
            emotional_keywords=emotional_keywords
        )
        
        # Store in memory
        self.memory.store_memory(similar_context)
        
        # Surprise should be lower for similar content
        self.assertLess(
            self.memory.surprise_scores[1],
            self.memory.surprise_scores[0]
        )
        
        # Get updated historical context
        new_memory_context = self.memory.get_historical_context(similar_context)
        
        # Process new emotional state
        new_state, new_context = self.emotional_processor(
            similar_context,
            new_memory_context
        )
        
        # States should be similar but not identical due to memory influence
        similarity = torch.cosine_similarity(emotional_state, new_state, dim=0)
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 1.0)
        
    def test_memory_hierarchy(self):
        """Test hierarchical memory storage based on surprise."""
        # Generate series of related and unrelated inputs
        inputs = [
            "I love spending time with my family.",
            "My family always makes me happy.",  # Related to first
            "I'm terrified of heights.",  # Unrelated
            "Being with my loved ones brings joy.",  # Related to first/second
            "Spiders really scare me."  # Related to third (fear)
        ]
        
        # Process through pipeline
        for text in inputs:
            context = self.contextualizer(text)
            self.memory.store_memory(context)
            
        # Verify surprise-based hierarchy
        scores = self.memory.surprise_scores
        
        # First input should have max surprise
        self.assertAlmostEqual(scores[0], 1.0)
        
        # Related inputs should have lower surprise
        self.assertLess(scores[1], scores[0])  # Family-related
        self.assertLess(scores[3], scores[0])  # Family-related
        
        # Unrelated inputs should have higher surprise
        self.assertGreater(scores[2], scores[1])  # Fear vs family
        
    def test_emotional_continuity(self):
        """Test emotional state continuity through memory integration."""
        # Process sequence of related emotional states
        sequence = [
            "I feel calm and peaceful.",
            "Everything is so serene.",
            "I'm in a state of tranquility."
        ]
        
        states = []
        for text in sequence:
            # Process through pipeline
            context = self.contextualizer(text)
            self.memory.store_memory(context)
            memory_context = self.memory.get_historical_context(context)
            state, _ = self.emotional_processor(context, memory_context)
            states.append(state)
            
        # Verify emotional continuity
        for i in range(1, len(states)):
            similarity = torch.cosine_similarity(states[i], states[i-1], dim=0)
            self.assertGreater(
                similarity,
                0.7,  # High similarity for related emotional states
                f"States {i} and {i-1} should be similar for related emotions"
            )

if __name__ == '__main__':
    unittest.main()
