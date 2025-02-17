"""Property-based tests for temporal parsing system."""

import pytest
from hypothesis import given, strategies as st, assume, settings
import time
from datetime import datetime, timedelta
import torch
from src.memory.temporal_parser import (
    TemporalParser,
    TemporalReference,
    OrdinalLink,
    TemporalContext
)

# Custom strategies for temporal expressions
@st.composite
def temporal_expressions(draw):
    """Generate valid temporal expressions."""
    patterns = [
        # Basic patterns
        st.just("yesterday"),
        st.just("last week"),
        st.just("last month"),
        # Life stages
        st.just("when I was a child"),
        st.just("during my teenage years"),
        # Emotional anchors
        st.just("the happiest time"),
        st.just("the worst moment"),
        # Frequencies
        st.just("every day"),
        st.just("occasionally"),
        # Cultural events
        st.just("last Christmas"),
        st.just("New Year's"),
    ]
    return draw(st.one_of(patterns))

@st.composite
def temporal_chains(draw):
    """Generate chains of temporal references."""
    base = draw(temporal_expressions())
    chain_length = draw(st.integers(min_value=1, max_value=5))
    relations = ["before", "after", "during", "around"]
    
    chain = [base]
    for _ in range(chain_length):
        relation = draw(st.sampled_from(relations))
        chain.append(f"{relation} that")
    
    return chain

@st.composite
def emotional_valences(draw):
    """Generate emotional valence values."""
    return draw(st.floats(min_value=-1.0, max_value=1.0))

class TestTemporalProperties:
    @pytest.fixture
    def parser(self):
        return TemporalParser()
    
    @given(temporal_expressions())
    def test_basic_parsing_properties(self, parser, expr):
        """Test basic properties that should hold for all temporal expressions."""
        ref = parser.parse_expression(expr)
        
        # Basic sanity checks
        assert ref.mean_time <= time.time()  # Can't be in future
        assert ref.uncertainty >= 0  # Uncertainty must be positive
        assert 0 <= ref.confidence <= 1  # Confidence must be in [0,1]
        
    @given(temporal_chains())
    def test_chain_uncertainty_monotonicity(self, parser, chain):
        """Test that uncertainty increases monotonically in reference chains."""
        refs = []
        for expr in chain:
            ref = parser.parse_expression(expr)
            refs.append(ref)
            
        # Check monotonic uncertainty increase
        for i in range(len(refs) - 1):
            assert refs[i+1].uncertainty >= refs[i].uncertainty
            
    @given(
        temporal_expressions(),
        emotional_valences()
    )
    def test_emotional_impact_consistency(self, parser, expr, valence):
        """Test consistent impact of emotional valence on temporal uncertainty."""
        # Reference without emotional valence
        ref_neutral = parser.parse_expression(expr)
        
        # Reference with emotional valence
        ref_emotional = parser.parse_expression(expr, emotional_valence=valence)
        
        # Strong emotions should reduce uncertainty
        if abs(valence) > 0.7:
            assert ref_emotional.uncertainty <= ref_neutral.uncertainty
            
    @given(st.lists(temporal_expressions(), min_size=1, max_size=5))
    def test_context_stability(self, parser, expressions):
        """Test that temporal context remains stable across multiple parses."""
        refs = []
        contexts = []
        
        for expr in expressions:
            ref = parser.parse_expression(expr)
            refs.append(ref)
            # Capture context state
            contexts.append(len(parser.context.recent_references))
            
        # Context should grow monotonically
        assert all(b >= a for a, b in zip(contexts, contexts[1:]))
        
    @given(
        temporal_expressions(),
        st.integers(min_value=1, max_value=365)
    )
    def test_temporal_distance_consistency(self, parser, expr, days_ago):
        """Test consistency of temporal distances."""
        now = time.time()
        
        # Create an anchor point
        anchor_time = now - (days_ago * 24 * 3600)
        anchor_ref = TemporalReference(
            mean_time=anchor_time,
            uncertainty=1.0,
            confidence=0.9,
            ordinal_relations=[],
            context_weight=0.8
        )
        parser.add_personal_anchor("test_anchor", anchor_ref)
        
        # Test relative references
        before_ref = parser.parse_expression(f"before test_anchor")
        after_ref = parser.parse_expression(f"after test_anchor")
        
        # Verify temporal ordering
        assert before_ref.mean_time < anchor_time
        assert after_ref.mean_time > anchor_time
        
    @given(st.lists(temporal_expressions(), min_size=2, max_size=10))
    @settings(max_examples=100)
    def test_memory_consistency(self, parser, expressions):
        """Test consistency of memory handling across multiple references."""
        refs = []
        memory_sizes = []
        
        for expr in expressions:
            ref = parser.parse_expression(expr)
            refs.append(ref)
            memory_sizes.append(len(parser.memories) if hasattr(parser, 'memories') else 0)
            
        # Memory size should never exceed max_memories
        assert all(size <= parser.max_memories for size in memory_sizes)
        
    @given(
        temporal_expressions(),
        st.lists(temporal_expressions(), min_size=1, max_size=3)
    )
    def test_ordinal_relation_consistency(self, parser, base_expr, related_exprs):
        """Test consistency of ordinal relationships."""
        base_ref = parser.parse_expression(base_expr)
        
        for expr in related_exprs:
            # Create an ordinal relationship
            link = OrdinalLink(
                target_ref=base_ref,
                relation_type="before",
                confidence=0.9,
                temporal_distance=30.0
            )
            
            # Add to context
            parser.context.recent_references.append(base_ref)
            
            # Parse related expression
            related_ref = parser.parse_expression(expr)
            
            # Verify relationship properties
            if hasattr(related_ref, 'ordinal_relations') and related_ref.ordinal_relations:
                relation = related_ref.ordinal_relations[0]
                assert relation.confidence <= base_ref.confidence
                assert relation.temporal_distance is not None
                
class TestComplexTemporalChains:
    """Tests for complex chains of temporal references."""
    
    @pytest.fixture
    def parser(self):
        return TemporalParser()
        
    def test_nested_temporal_references(self, parser):
        """Test handling of nested temporal references."""
        # Set up a complex chain of events
        expressions = [
            "during college",
            "right after that difficult period",
            "before I realized what was happening",
            "around the time everything changed"
        ]
        
        refs = []
        for expr in expressions:
            ref = parser.parse_expression(expr)
            refs.append(ref)
            parser.context.recent_references.append(ref)
            
        # Verify increasing uncertainty
        uncertainties = [ref.uncertainty for ref in refs]
        assert all(b > a for a, b in zip(uncertainties, uncertainties[1:]))
        
        # Verify decreasing confidence
        confidences = [ref.confidence for ref in refs]
        assert all(b < a for a, b in zip(confidences, confidences[1:]))
        
    def test_emotional_temporal_sequence(self, parser):
        """Test sequence of emotionally anchored temporal references."""
        sequence = [
            ("the happiest day", 0.9),
            ("when everything fell apart", -0.8),
            ("as I started to recover", 0.3),
            ("when I finally felt like myself again", 0.7)
        ]
        
        refs = []
        for expr, valence in sequence:
            ref = parser.parse_expression(expr, emotional_valence=valence)
            refs.append(ref)
            parser.context.recent_references.append(ref)
            
        # Verify temporal ordering
        times = [ref.mean_time for ref in refs]
        assert all(b > a for a, b in zip(times, times[1:]))
        
    def test_mixed_temporal_cultural_chain(self, parser):
        """Test chain mixing cultural events and personal references."""
        # Set up cultural anchors
        christmas_time = datetime(datetime.now().year - 1, 12, 25).timestamp()
        new_year_time = datetime(datetime.now().year, 1, 1).timestamp()
        
        sequence = [
            "last Christmas",
            "right after the holidays",
            "when I made my resolution",
            "as I started working towards my goals"
        ]
        
        refs = []
        for expr in sequence:
            ref = parser.parse_expression(expr)
            refs.append(ref)
            
        # Verify reasonable temporal progression
        assert abs(refs[0].mean_time - christmas_time) < 24*3600  # Within a day
        assert refs[1].mean_time > christmas_time
        assert abs(refs[2].mean_time - new_year_time) < 7*24*3600  # Within a week
        
    def test_frequency_based_chain(self, parser):
        """Test chain of frequency-based references."""
        sequence = [
            "every day last week",
            "occasionally this month",
            "rarely nowadays"
        ]
        
        refs = []
        for expr in sequence:
            ref = parser.parse_expression(expr)
            refs.append(ref)
            
        # Verify frequency properties
        assert len(refs[0].ordinal_relations) > 0  # Regular pattern
        assert refs[1].uncertainty > refs[0].uncertainty  # Less certain
        assert refs[2].uncertainty > refs[1].uncertainty  # Even less certain
        
    def test_life_stage_progression(self, parser):
        """Test progression through life stages."""
        stages = [
            "when I was very young",
            "during elementary school",
            "in my teenage years",
            "during college",
            "in my early twenties"
        ]
        
        refs = []
        for expr in stages:
            ref = parser.parse_expression(expr)
            refs.append(ref)
            
        # Verify chronological ordering
        times = [ref.mean_time for ref in refs]
        assert all(b > a for a, b in zip(times, times[1:]))
        
        # Verify uncertainty patterns
        uncertainties = [ref.uncertainty for ref in refs]
        # Earlier memories should have higher uncertainty
        assert all(b < a for a, b in zip(uncertainties, uncertainties[1:]))
