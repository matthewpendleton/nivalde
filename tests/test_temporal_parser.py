"""Unit tests for temporal parsing system."""

import pytest
import time
from datetime import datetime, timedelta
import torch
from src.memory.temporal_parser import (
    TemporalParser,
    TemporalReference,
    OrdinalLink,
    TemporalContext
)

class TestTemporalParser:
    @pytest.fixture
    def parser(self):
        return TemporalParser()
        
    @pytest.fixture
    def now(self):
        return time.time()
        
    def test_basic_temporal_patterns(self, parser, now):
        """Test basic temporal expressions."""
        cases = [
            ("yesterday", 1, 0.2),
            ("last week", 7, 2.0),
            ("last month", 30, 7.0),
            ("few months ago", 90, 30.0),
        ]
        
        for expr, expected_days, expected_uncertainty in cases:
            ref = parser.parse_expression(expr)
            expected_time = now - (expected_days * 24 * 3600)
            assert abs(ref.mean_time - expected_time) < 5  # Allow 5 second tolerance
            assert abs(ref.uncertainty - expected_uncertainty) < 0.1
            assert ref.confidence > 0.7
            
    def test_life_stage_patterns(self, parser):
        """Test life stage temporal expressions."""
        cases = [
            ("when I was a child", 7300, 1825.0, 0.7),
            ("during my teenage years", 5475, 1095.0, 0.7),
            ("in college", 3650, 730.0, 0.8),
        ]
        
        for expr, expected_days, expected_uncertainty, expected_conf in cases:
            ref = parser.parse_expression(expr)
            assert ref.uncertainty == pytest.approx(expected_uncertainty, rel=0.1)
            assert ref.confidence == pytest.approx(expected_conf, rel=0.1)
            
    def test_emotional_anchoring(self, parser):
        """Test emotionally anchored temporal expressions."""
        # Set up an emotional peak
        peak_time = time.time() - (180 * 24 * 3600)  # 6 months ago
        peak_ref = TemporalReference(
            mean_time=peak_time,
            uncertainty=7.0,
            confidence=0.9,
            ordinal_relations=[],
            context_weight=0.8
        )
        parser.add_emotional_peak("positive_peak", peak_ref)
        
        # Test reference to the peak
        ref = parser.parse_expression(
            "during the happiest time",
            emotional_valence=0.8
        )
        assert ref.mean_time == pytest.approx(peak_time, rel=0.001)
        assert ref.uncertainty < peak_ref.uncertainty  # Should be more certain with high valence
        assert ref.confidence > 0.8
        
    def test_frequency_patterns(self, parser):
        """Test frequency-based temporal expressions."""
        cases = [
            # (expression, period, uncertainty, is_regular)
            ("every day", 1, 0.2, True),
            ("weekly", 7, 1.0, True),
            ("occasionally", 14, 7.0, False),
            ("rarely", 90, 30.0, False),
        ]
        
        for expr, period, uncertainty, is_regular in cases:
            ref = parser.parse_expression(expr)
            assert ref.uncertainty == pytest.approx(
                uncertainty if not is_regular else uncertainty,
                rel=0.1
            )
            if is_regular:
                assert len(ref.ordinal_relations) > 0  # Should have links to previous occurrences
                
    def test_cultural_events(self, parser):
        """Test cultural and personal event references."""
        # Set up a personal milestone
        wedding_time = time.time() - (365 * 24 * 3600)  # 1 year ago
        wedding_ref = TemporalReference(
            mean_time=wedding_time,
            uncertainty=1.0,
            confidence=0.95,
            ordinal_relations=[],
            context_weight=0.9
        )
        parser.add_personal_anchor("wedding", wedding_ref)
        
        # Test fixed calendar event
        christmas_ref = parser.parse_expression("last Christmas")
        expected_christmas = datetime(datetime.now().year - 1, 12, 25).timestamp()
        assert christmas_ref.mean_time == pytest.approx(expected_christmas, rel=0.1)
        assert christmas_ref.uncertainty == pytest.approx(5.0, rel=0.1)
        
        # Test personal milestone reference
        ref = parser.parse_expression("during my wedding")
        assert ref.mean_time == pytest.approx(wedding_time, rel=0.001)
        assert ref.uncertainty < 2.0  # Should be quite certain
        assert ref.confidence > 0.9
        
    def test_temporal_clustering(self, parser):
        """Test temporal clustering expressions."""
        # Set up a base reference
        base_time = time.time() - (10 * 24 * 3600)  # 10 days ago
        base_ref = TemporalReference(
            mean_time=base_time,
            uncertainty=1.0,
            confidence=0.9,
            ordinal_relations=[],
            context_weight=0.8
        )
        parser.context.recent_references.append(base_ref)
        
        # Test concurrent events
        concurrent_ref = parser.parse_expression("at that same time")
        assert concurrent_ref.mean_time == pytest.approx(base_time, rel=0.001)
        assert concurrent_ref.uncertainty < base_ref.uncertainty  # Should be more certain
        assert len(concurrent_ref.ordinal_relations) == 1
        assert concurrent_ref.ordinal_relations[0].relation_type == "during"
        
        # Test immediate sequence
        sequence_ref = parser.parse_expression("right after that")
        assert sequence_ref.mean_time > base_time
        assert sequence_ref.uncertainty > base_ref.uncertainty  # Should be less certain
        assert len(sequence_ref.ordinal_relations) == 1
        assert sequence_ref.ordinal_relations[0].relation_type == "after"
        
    def test_ordinal_relations(self, parser):
        """Test ordinal relationship handling."""
        # Set up anchor points
        graduation_time = time.time() - (730 * 24 * 3600)  # 2 years ago
        grad_ref = TemporalReference(
            mean_time=graduation_time,
            uncertainty=1.0,
            confidence=0.95,
            ordinal_relations=[],
            context_weight=0.9
        )
        parser.add_personal_anchor("graduation", grad_ref)
        
        cases = [
            ("before my graduation", "before", -30),
            ("after graduation", "after", 30),
            ("during graduation", "during", 0),
            ("around graduation", "around", 0),
        ]
        
        for expr, expected_relation, expected_offset in cases:
            ref = parser.parse_expression(expr)
            assert len(ref.ordinal_relations) == 1
            assert ref.ordinal_relations[0].relation_type == expected_relation
            if expected_offset:
                time_diff = (ref.mean_time - graduation_time) / (24 * 3600)
                assert abs(time_diff - expected_offset) < 5  # Within 5 days
                
    def test_uncertainty_propagation(self, parser):
        """Test how uncertainty propagates through temporal relationships."""
        base_ref = parser.parse_expression("last month")  # Start with known uncertainty
        
        # Sequence of increasingly uncertain references
        refs = [
            parser.parse_expression("before that"),
            parser.parse_expression("sometime before that"),
            parser.parse_expression("long before that")
        ]
        
        # Uncertainty should increase with each step
        for i in range(len(refs) - 1):
            assert refs[i+1].uncertainty > refs[i].uncertainty
            assert refs[i+1].confidence < refs[i].confidence
            
    def test_emotional_valence_impact(self, parser):
        """Test how emotional valence affects temporal uncertainty."""
        cases = [
            (0.9, "the happiest moment"),  # High positive valence
            (0.1, "a difficult time"),     # Low positive valence
            (-0.8, "the worst day"),       # High negative valence
            (-0.2, "a bit sad"),           # Low negative valence
        ]
        
        refs = [
            parser.parse_expression(expr, emotional_valence=valence)
            for valence, expr in cases
        ]
        
        # Strong emotions (both positive and negative) should reduce uncertainty
        assert refs[0].uncertainty < refs[1].uncertainty  # High positive vs low positive
        assert refs[2].uncertainty < refs[3].uncertainty  # High negative vs low negative
        
    def test_context_influence(self, parser):
        """Test how context influences temporal parsing."""
        # Set up a strong context
        context_time = time.time() - (90 * 24 * 3600)  # 3 months ago
        context_ref = TemporalReference(
            mean_time=context_time,
            uncertainty=5.0,
            confidence=0.9,
            ordinal_relations=[],
            context_weight=0.9
        )
        parser.context.add_anchor("moving", context_ref)
        
        # Test how context affects vague expressions
        ref = parser.parse_expression("around when I moved")
        assert ref.mean_time == pytest.approx(context_time, rel=0.001)
        assert ref.uncertainty < 10.0  # Should be more certain with context
        assert ref.context_weight > 0.7  # Should rely heavily on context
