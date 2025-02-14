import unittest
import torch
import numpy as np
import sounddevice as sd
from src.input.multimodal_processor import MultimodalProcessor
from src.memory.emotional_embedding import EmotionalProcessor
from src.therapy.latent_manifold import TherapeuticPlanner

class TestTherapyPipeline(unittest.TestCase):
    def setUp(self):
        # Initialize components with consistent embedding dimensions
        self.multimodal_processor = MultimodalProcessor(
            audio_embedding_dim=256,
            contextual_dim=768
        )
        self.emotional_processor = EmotionalProcessor(embedding_dim=768)
        self.therapeutic_planner = TherapeuticPlanner(embedding_dim=768)
        
    def generate_mock_audio(self, duration_seconds: float = 1.0, sample_rate: int = 44100):
        """Generate mock audio data (simulated emotional speech)"""
        # Create a mixture of frequencies to simulate speech
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        
        # Emotional speech typically has frequencies between 100-500 Hz
        frequencies = [150, 250, 350]
        amplitudes = [0.5, 0.3, 0.2]
        
        # Generate complex waveform
        audio = np.zeros_like(t)
        for f, a in zip(frequencies, amplitudes):
            audio += a * np.sin(2 * np.pi * f * t)
            
        # Add some noise to make it more realistic
        audio += np.random.normal(0, 0.1, len(t))
        
        return torch.from_numpy(audio.astype(np.float32))
        
    def test_full_pipeline(self):
        """Test the complete therapy pipeline from input to intervention"""
        # 1. Generate mock audio input
        audio_data = self.generate_mock_audio()
        
        # 2. Process multimodal input (audio only for this test)
        multimodal_output = self.multimodal_processor.process_input(
            audio_data=audio_data
        )
        
        # Verify multimodal processing
        self.assertIsNotNone(multimodal_output.audio_embedding)
        self.assertIsNotNone(multimodal_output.contextual_embedding)
        self.assertEqual(multimodal_output.contextual_embedding.size(-1), 768)
        
        # 3. Process through emotional embedding space
        emotional_state, transitions = self.emotional_processor.process(
            multimodal_output.contextual_embedding
        )
        
        # Verify emotional processing
        self.assertEqual(emotional_state.size(-1), 768)
        self.assertIsInstance(transitions, list)
        
        # 4. Generate therapeutic intervention
        intervention = self.therapeutic_planner.plan_intervention(
            emotional_state,
            temperature=0.8  # Slightly lower temperature for more focused interventions
        )
        
        # Verify intervention
        self.assertEqual(intervention.size(-1), 768)
        
        # 5. Test memory retention
        # Process same input again to test memory effects
        emotional_state_2, transitions_2 = self.emotional_processor.process(
            multimodal_output.contextual_embedding
        )
        
        # Memory should influence the emotional state
        self.assertFalse(
            torch.allclose(emotional_state, emotional_state_2),
            "Memory should influence emotional state processing"
        )
        
    def test_phase_transitions(self):
        """Test detection of therapeutic opportunities/risks"""
        # Generate two different emotional states
        audio_1 = self.generate_mock_audio(duration_seconds=1.0)
        audio_2 = self.generate_mock_audio(duration_seconds=1.0)
        
        # Process first state
        output_1 = self.multimodal_processor.process_input(audio_data=audio_1)
        emotional_state_1, _ = self.emotional_processor.process(
            output_1.contextual_embedding
        )
        
        # Process second state
        output_2 = self.multimodal_processor.process_input(audio_data=audio_2)
        emotional_state_2, transitions = self.emotional_processor.process(
            output_2.contextual_embedding
        )
        
        # Verify phase transition detection
        self.assertIsInstance(transitions, list)
        
    def test_intervention_optimization(self):
        """Test RL-based optimization of interventions"""
        # Generate initial state
        audio_data = self.generate_mock_audio()
        multimodal_output = self.multimodal_processor.process_input(
            audio_data=audio_data
        )
        emotional_state, _ = self.emotional_processor.process(
            multimodal_output.contextual_embedding
        )
        
        # Generate multiple interventions
        interventions = [
            self.therapeutic_planner.plan_intervention(emotional_state)
            for _ in range(3)
        ]
        
        # Record mock outcomes
        for intervention in interventions:
            self.therapeutic_planner.record_outcome(
                emotional_state,
                intervention,
                outcome_score=np.random.random()  # Mock outcome score
            )
            
        # Generate optimized intervention
        optimized_intervention = self.therapeutic_planner.plan_intervention(
            emotional_state
        )
        
        # Verify optimization
        self.assertEqual(optimized_intervention.size(-1), 768)
        
if __name__ == '__main__':
    unittest.main()
